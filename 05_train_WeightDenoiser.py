import logging
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import argparse
import torch.nn.functional as F
import math

# 假设这些是你自定义的项目模块
from battery_data import BatteryData
from model import GMANet, GMANetPreTrain, WeightDenoiser, FlowMatchingDenoiser 

W_DIM = 64 * 32

# ==============================================================================
# 1. 扩散调度类 (Simple DDPM Scheduler)
# 该类负责管理加噪（Forward）和去噪（Reverse）的数学过程
# ==============================================================================

class DiffusionManager:
    def __init__(self, T=400, device='cuda'):
        self.T = T  # 总扩散步数
        self.device = device
        
        # 线性 Beta 调度：控制每个时间步注入噪声的比例 (从万分之一到百分之二)
        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        
        # 预计算推导扩散过程所需的系数 (基于 DDPM 论文公式)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # alpha_bar: 累乘
        
    def q_sample(self, x_0, t, noise=None):
        """ 
        前向扩散过程：在给定 x_0 和时间步 t 的情况下，一步生成加噪后的 x_t
        公式：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # 提取对应时间步的系数并调整形状以适配 batch 运算
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def q_sample_with_anchor(self, x_0, x_anchor, t, noise=None):
        """
        以锚点向量 x_anchor 为扩散起点进行前向加噪：
        x_t = x_anchor + sqrt(alpha_bar_t) * (x_0 - x_anchor) + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1)
        residual = x_0 - x_anchor
        return x_anchor + sqrt_alphas_cumprod_t * residual + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample_loop(self, model, global_feat, local_feat, shape, init_vec=None):
        """ 
        逆向去噪过程（采样）：从纯高斯噪声开始，通过模型预测的噪声逐步还原出 x_0
        """
        model.eval()
        # 逆扩散起点：优先使用预训练 GMA 的 A/B/S 原型向量，否则回退随机噪声
        if init_vec is not None:
            cur_x = init_vec.to(self.device).view(1, -1).repeat(shape[0], 1)
        else:
            cur_x = torch.randn(shape, device=self.device)
        
        # 从 T-1 步倒数到 0 步进行迭代
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            
            # 模型 WeightDenoiser 预测当前 cur_x 中包含的噪声
            predicted_noise = model(cur_x, t, global_feat, local_feat)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            # 只有当 i > 0 时才添加随机扰动，最后一步 (i=0) 得到确定性结果
            noise = torch.randn_like(cur_x) if i > 0 else 0
            
            # 基于预测噪声计算前一个状态的均值 (DDPM 采样公式)
            mean = (1 / torch.sqrt(alpha_t)) * (
                cur_x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )
            
            cur_x = mean + torch.sqrt(beta_t) * noise
            
            # 可选：数值截断，防止推理时产生离群点导致反归一化爆炸
            cur_x = torch.clamp(cur_x, -5.0, 5.0) 
            
        return cur_x


class FlowMatchingManager:
    def __init__(self, T=400, device='cuda'):
        self.T = T
        self.device = device

    @torch.no_grad()
    def sample(self, model, global_feat, local_feat, shape, init_vec=None):
        """ODE Euler sampler for strict flow matching."""
        model.eval()
        if init_vec is None:
            cur_x = torch.randn(shape, device=self.device)
        else:
            cur_x = init_vec.to(self.device)
            if cur_x.dim() == 1:
                cur_x = cur_x.unsqueeze(0)
        dt = 1.0 / max(self.T, 1)

        for i in range(self.T):
            t_value = i / max(self.T, 1)
            t = torch.full((shape[0],), t_value, device=self.device, dtype=torch.float32)
            velocity = model(cur_x, t, global_feat, local_feat)
            cur_x = cur_x + dt * velocity
            cur_x = torch.clamp(cur_x, -5.0, 5.0)

        return cur_x

# 数据载入辅助类
class HyperCacheDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list # 存储 (target_vec, feat_l, feat_g) 的元组列表
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

# ==============================================================================
# 2. 基础工具函数：模型加载与数据向量化
# ==============================================================================

def load_frozen_encoder(pretrained_path, device):
    """ 加载预训练的电池特征提取器（编码器），并冻结参数 """
    model = GMANetPreTrain().to(device)
    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    encoder = model.encoder
    encoder.eval()
    for p in encoder.parameters(): 
        p.requires_grad = False
    return encoder


def load_frozen_soh_evaluator(pretrained_path, device):
    """加载冻结的 SOH 评估器，用于训练时的直达 SOH 约束。"""
    model = GMANet().to(device)
    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def build_X_from_cycles(cycles, frozen_encoder, device):
    """ 将原始电池循环数据（电压/容量曲线）输入编码器，转换为高维特征向量 """
    cycles_tensor = torch.from_numpy(cycles).float() if isinstance(cycles, np.ndarray) else cycles.float()
    frozen_encoder.eval()
    with torch.no_grad():
        # 输出形状通常为 [Sequence_Length, Hidden_Dim]，建议在后面做平均池化简化
        feat = frozen_encoder(cycles_tensor.to(device))
    return feat.cpu()

def load_data_flattened(weight_path, device, train_dir, local_num, global_num, eps=1e-8):
    """
    数据预处理核心逻辑：
    1. 直接使用窗口权重 W(64x32)
    2. 展平为 2048 维向量，作为 Diff/Flow 的生成目标
    """
    weights = torch.load(weight_path, map_location=device)
    if "weight_W" in weights:
        target_vec = weights["weight_W"].to(device).float().view(-1)
    else:
        # 向后兼容旧权重文件：由 A/B/log_s 还原 W
        p_A, p_B = weights["param_A"].to(device).float(), weights["param_B"].to(device).float()
        if "param_log_s" in weights:
            p_log_s = weights["param_log_s"].to(device).float().view(1)
        else:
            norm_val = torch.norm(p_A, p="fro") * torch.norm(p_B, p="fro")
            p_log_s = torch.log(norm_val + eps).view(1)
        scale = torch.exp(torch.clamp(p_log_s, min=-5.0, max=5.0)).view(1, 1)
        target_vec = (scale * torch.matmul(p_A, p_B)).contiguous().view(-1)

    # 根据文件名查找对应的电池原始数据文件
    parts = weight_path.stem.split("_")
    start_idx = int(parts[-1]) # 权重对应的起始循环索引
    battery_name = "_".join(parts[:-1])
    battery_path = train_dir / f"{battery_name}.pkl"
    battery = BatteryData.load(str(battery_path))

    # 提取全局（电池初期）和局部（当前时刻）循环数据
    g_cyc = np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in battery.cycle_data[0:global_num]])
    l_cyc = np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in battery.cycle_data[start_idx:start_idx+local_num]])

    return target_vec.cpu(), torch.tensor(l_cyc), torch.tensor(g_cyc)


def load_target_soh(weight_path, train_dir, window_size):
    """根据权重文件名定位对应电池，并取与评估一致的完整窗口 SOH 作为监督信号。"""
    parts = weight_path.stem.split("_")
    start_idx = int(parts[-1])
    battery_name = "_".join(parts[:-1])
    battery_path = train_dir / f"{battery_name}.pkl"
    battery = BatteryData.load(str(battery_path))

    end_idx = min(start_idx + window_size, len(battery.cycle_data))
    window_cycles = battery.cycle_data[start_idx:end_idx]
    target_soh = [cycle.labeled_soh for cycle in window_cycles]
    if not target_soh or any(soh is None for soh in target_soh):
        raise ValueError(f"Missing labeled_soh in {battery_path} for window [{start_idx}, {end_idx})")
    return torch.tensor(target_soh, dtype=torch.float32)


def load_window_cycles(weight_path, train_dir, window_size):
    """Load the exact window cycles used by the end-to-end path."""
    parts = weight_path.stem.split("_")
    start_idx = int(parts[-1])
    battery_name = "_".join(parts[:-1])
    battery_path = train_dir / f"{battery_name}.pkl"
    battery = BatteryData.load(str(battery_path))

    end_idx = min(start_idx + window_size, len(battery.cycle_data))
    window_cycles = battery.cycle_data[start_idx:end_idx]
    return torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in window_cycles]), dtype=torch.float32)


def load_pretrained_abs_init(init_abs_path, device):
    """加载预训练初始化并转换为 2048 维 W 向量。"""
    if not init_abs_path.exists():
        logging.warning(f'ABS init file not found: {init_abs_path}. Fallback to random noise.')
        return None

    data = torch.load(init_abs_path, map_location=device)
    if 'weight_W' in data:
        init_vec = data['weight_W'].float().view(-1)
    elif 'init_vec' in data and data['init_vec'].numel() == W_DIM:
        init_vec = data['init_vec'].float().view(-1)
    elif 'param_A' in data and 'param_B' in data and 'param_log_s' in data:
        p_A = data['param_A'].float()
        p_B = data['param_B'].float()
        p_log_s = data['param_log_s'].float().view(1)
        scale = torch.exp(torch.clamp(p_log_s, min=-5.0, max=5.0)).view(1, 1)
        init_vec = (scale * torch.matmul(p_A, p_B)).contiguous().view(-1)
    else:
        logging.warning(f'Invalid ABS init format: {init_abs_path}. Fallback to random noise.')
        return None

    if init_vec.numel() != W_DIM:
        logging.warning(f'ABS init size mismatch ({init_vec.numel()}). Expected {W_DIM}. Fallback to random noise.')
        return None

    return init_vec.to(device)

# ==============================================================================
# 3. 核心训练逻辑
# ==============================================================================
# def train_diffusion(
#     train_loader,
#     model_save_path,
#     device,
#     diffusion_steps,
#     denoiser_hidden_dim,
#     denoiser_type,
#     window_size,
#     target_mean,
#     target_std,
#     init_vec,
#     encoder,
#     soh_evaluator,
#     epochs=100000,
#     use_soh_loss=True,
#     use_log_s_reg=True,
#     use_dynamic_weights=True,
# ):
#     # 扩散步数与网络宽度需和评估保持一致
#     diff_manager = DiffusionManager(T=diffusion_steps, device=device)

#     # 定义去噪网络：输入 (噪声向量, 时间步, 全局特征, 局部特征)
#     denoiser_cls = WeightDenoiser if denoiser_type == "diff" else FlowMatchingDenoiser
#     model = denoiser_cls(weight_dim=W_DIM, hidden_dim=denoiser_hidden_dim, cond_input_dim=cond_input_dim).to(device)
    
#     # 移动目标统计值到设备
#     target_mean = target_mean.to(device)
#     target_std = target_std.to(device)

#     if init_vec is None:
#         raise ValueError("GMA init vector is required. Please make sure results/GMA_pretrained_abs.pth exists.")
#     # 训练目标在标准化空间，锚点向量也必须标准化后再参与扩散
#     init_vec_norm = ((init_vec.to(device).view(1, -1) - target_mean.view(1, -1)) / (target_std.view(1, -1) + 1e-8))
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=50
#     )
    
#     # 用于早停与 alpha 切换的变量
#     best_loss = float('inf')
#     plateau_counter = 0              # 连续未创新低的 epoch 数（用于 alpha 切换）
#     early_stop_counter = 0           # 连续未创新低的 epoch 数（用于早停）
#     alpha = 0.0                      # 0：基础阶段，1：下游主导阶段
#     alpha_locked = False             # 一旦激活 alpha=1，不再回退

#     logging.info("--- 开始扩散模型增强训练 ---")
    
#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0

#         # 根据当前的 alpha 计算损失权重（alpha 为 0 或 1）
#         if use_dynamic_weights:
#             if alpha < 1.0:
#                 # 基础阶段：噪声/向量主导
#                 noise_w, vec_w, log_s_w, w_w, soh_w = 1.0, 0.2, 0.5, 0.2, 0.8
#             else:
#                 # 下游主导阶段：W/SOH 主导（注：w_w 此处未实际使用，可忽略）
#                 noise_w, vec_w, log_s_w, w_w, soh_w = 0.3, 0.05, 0.75, 2.0, 15.0
#         else:
#             noise_w, vec_w, log_s_w, w_w, soh_w = 1.0, 0.2, 0.5, 0.2, 0.8

#         for target_vec, feat_l, feat_g, w_cyc, target_soh in train_loader:
#             target_vec = target_vec.to(device)
#             feat_l = feat_l.to(device)
#             feat_g = feat_g.to(device)
#             w_cyc = w_cyc.to(device)
#             target_soh = target_soh.to(device)
            
#             optimizer.zero_grad()

#             if denoiser_type == "diff":
#                 t = torch.randint(0, diff_manager.T, (target_vec.shape[0],), device=device).long()
#                 noise = torch.randn_like(target_vec)
#                 anchor_vec = init_vec_norm.expand(target_vec.shape[0], -1)
#                 z_t = diff_manager.q_sample_with_anchor(target_vec, anchor_vec, t, noise)
#                 predicted_signal = model(z_t, t, feat_g, feat_l)
#                 signal_loss_map = F.mse_loss(predicted_signal, noise, reduction='none')
#                 signal_loss_map[:, -1] *= 20.0
#                 base_loss = signal_loss_map.mean()
#                 sqrt_ab_t = torch.sqrt(diff_manager.alphas_cumprod[t]).view(-1, 1)
#                 sqrt_one_minus_ab_t = torch.sqrt(1 - diff_manager.alphas_cumprod[t]).view(-1, 1)
#                 pred_x0 = anchor_vec + (z_t - anchor_vec - sqrt_one_minus_ab_t * predicted_signal) / (sqrt_ab_t + 1e-8)
#             else:
#                 base_sample = init_vec_norm.expand(target_vec.shape[0], -1)
#                 t = torch.rand(target_vec.shape[0], device=device).clamp(1e-4, 1.0 - 1e-4)
#                 t_view = t.view(-1, 1)
#                 z_t = (1.0 - t_view) * base_sample + t_view * target_vec
#                 target_flow = target_vec - base_sample
#                 predicted_signal = model(z_t, t, feat_g, feat_l)
#                 signal_loss_map = F.mse_loss(predicted_signal, target_flow, reduction='none')
#                 signal_loss_map[:, -1] *= 20.0
#                 base_loss = signal_loss_map.mean()
#                 pred_x0 = z_t + (1.0 - t_view) * predicted_signal

#             vec_loss = F.mse_loss(pred_x0[:, :384], target_vec[:, :384])
#             pred_log_s = pred_x0[:, 384:385]
#             target_log_s = target_vec[:, 384:385]
#             log_s_fit_loss = F.smooth_l1_loss(pred_log_s, target_log_s, beta=0.1)
#             if use_log_s_reg:
#                 log_s_mean = target_log_s.mean().detach()
#                 log_s_std = target_log_s.std().detach().clamp_min(1e-8)
#                 log_s_reg_loss = torch.mean(((pred_log_s - log_s_mean) / log_s_std) ** 2)
#                 log_s_loss = log_s_fit_loss + 0.3 * log_s_reg_loss
#             else:
#                 log_s_loss = log_s_fit_loss

#             # 在下游实际使用的 W 空间做一致性约束（w_loss 计算但不参与 loss）
#             pred_x0_raw = pred_x0 * target_std + target_mean
#             target_x0_raw = target_vec * target_std + target_mean

#             p_A = pred_x0_raw[:, 0:256].view(-1, 64, 4)
#             p_B = pred_x0_raw[:, 256:384].view(-1, 4, 32)
#             p_log_s = torch.clamp(pred_x0_raw[:, 384:385], min=-5.0, max=5.0)

#             t_A = target_x0_raw[:, 0:256].view(-1, 64, 4)
#             t_B = target_x0_raw[:, 256:384].view(-1, 4, 32)
#             t_log_s = torch.clamp(target_x0_raw[:, 384:385], min=-5.0, max=5.0)

#             pred_W = torch.exp(p_log_s).view(-1, 1, 1) * torch.bmm(p_A, p_B)
#             target_W = torch.exp(t_log_s).view(-1, 1, 1) * torch.bmm(t_A, t_B)
#             w_loss = F.mse_loss(pred_W, target_W)  # 可用于监控，但不参与反向传播

#             soh_loss = torch.tensor(0.0, device=device)
#             if use_soh_loss:
#                 feat_w = build_X_from_cycles(w_cyc.reshape(-1, w_cyc.shape[-1]), encoder, device)
#                 feat_w = feat_w.view(w_cyc.shape[0], w_cyc.shape[1], -1).to(device)
#                 pred_soh = soh_evaluator.apply_lora(
#                     feat_w.reshape(-1, feat_w.shape[-1]),
#                     p_A.repeat_interleave(feat_w.shape[1], dim=0),
#                     p_B.repeat_interleave(feat_w.shape[1], dim=0),
#                     p_log_s.repeat_interleave(feat_w.shape[1], dim=0),
#                 )
#                 soh_loss = F.smooth_l1_loss(pred_soh.view_as(target_soh), target_soh, beta=0.02)

#             # 组合总损失（当前版本不包含 w_loss）
#             loss = noise_w * base_loss + vec_w * vec_loss + log_s_w * log_s_loss + soh_w * soh_loss
#             # loss = noise_w * base_loss + vec_w * vec_loss + log_s_w * log_s_loss + w_w * w_loss + soh_w * soh_loss
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             epoch_loss += loss.item()

#         avg_loss = epoch_loss / len(train_loader)
#         scheduler.step(avg_loss)
        
#         # ----- 更新 best_loss、保存模型、更新计数器 -----
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save(model.state_dict(), model_save_path)  # 保存当前最优模型
#             logging.info("Best model so far.")
#             plateau_counter = 0
#             early_stop_counter = 0
#         else:
#             plateau_counter += 1
#             early_stop_counter += 1

#         # ----- alpha 切换逻辑 -----
#         if not alpha_locked and plateau_counter >= 100:
#             alpha = 1.0
#             alpha_locked = True
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = 2e-4
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, mode='min', factor=0.5, patience=50
#             )
#             logging.info('Plateau for 100 epochs, alpha locked to 1, LR reset to 2e-4')

#         # ----- 早停 -----
#         if early_stop_counter >= 300:
#             logging.info('Early stopping triggered after 300 epochs without improvement.')
#             break

#         logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | alpha: {alpha:.1f} | W: noise={noise_w:.2f} vec={vec_w:.2f} log_s={log_s_w:.2f} SOH={soh_w:.2f}")
def train_diffusion(
    train_loader,
    model_save_path,
    device,
    diffusion_steps,
    denoiser_hidden_dim,
    denoiser_type,
    window_size,
    target_mean,
    target_std,
    init_vec,
    encoder,
    soh_evaluator,
    cond_input_dim=100,
    epochs=100000,
    use_soh_loss=True,
    use_log_s_reg=True,
    use_dynamic_weights=True,
):
    # 扩散步数与网络宽度需和评估保持一致
    diff_manager = DiffusionManager(T=diffusion_steps, device=device)

    # 定义去噪网络
    denoiser_cls = WeightDenoiser if denoiser_type == "diff" else FlowMatchingDenoiser
    model = denoiser_cls(weight_dim=W_DIM, hidden_dim=denoiser_hidden_dim, cond_input_dim=cond_input_dim).to(device)
    
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    if init_vec is None:
        raise ValueError("GMA init vector is required.")
    init_vec_norm = ((init_vec.to(device).view(1, -1) - target_mean.view(1, -1)) / (target_std.view(1, -1) + 1e-8))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    best_loss = float('inf')
    plateau_counter = 0          # 连续未创新低 epoch 数（用于切换到 SOH 阶段）
    early_stop_counter = 0       # 连续未创新低 epoch 数（用于早停）
    in_soh_phase = False         # 是否已进入 SOH 主导阶段
    soh_phase_activated = False  # 确保只激活一次

    logging.info("--- 开始扩散模型增强训练 ---")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # 累积梯度强度统计
        sum_soh_grad_norm = 0.0
        sum_total_grad_norm = 0.0
        grad_count = 0

        # 根据阶段确定损失权重
        if use_dynamic_weights:
            if in_soh_phase:
                # SOH 主导阶段
                noise_w, w_w, soh_w = 0.3, 2.0, 10.0
            else:
                # 基础阶段
                noise_w, w_w, soh_w = 1.0, 0.5, 0.8
        else:
            noise_w, w_w, soh_w = 1.0, 0.5, 0.8

        for target_vec, feat_l, feat_g, w_cyc, target_soh in train_loader:
            target_vec = target_vec.to(device)
            feat_l = feat_l.to(device)
            feat_g = feat_g.to(device)
            w_cyc = w_cyc.to(device)
            target_soh = target_soh.to(device)

            optimizer.zero_grad()

            if denoiser_type == "diff":
                t = torch.randint(0, diff_manager.T, (target_vec.shape[0],), device=device).long()
                noise = torch.randn_like(target_vec)
                anchor_vec = init_vec_norm.expand(target_vec.shape[0], -1)
                z_t = diff_manager.q_sample_with_anchor(target_vec, anchor_vec, t, noise)
                predicted_signal = model(z_t, t, feat_g, feat_l)
                base_loss = F.mse_loss(predicted_signal, noise)
                sqrt_ab_t = torch.sqrt(diff_manager.alphas_cumprod[t]).view(-1, 1)
                sqrt_one_minus_ab_t = torch.sqrt(1 - diff_manager.alphas_cumprod[t]).view(-1, 1)
                pred_x0 = anchor_vec + (z_t - anchor_vec - sqrt_one_minus_ab_t * predicted_signal) / (sqrt_ab_t + 1e-8)
            else:
                base_sample = init_vec_norm.expand(target_vec.shape[0], -1)
                t = torch.rand(target_vec.shape[0], device=device).clamp(1e-4, 1.0 - 1e-4)
                t_view = t.view(-1, 1)
                z_t = (1.0 - t_view) * base_sample + t_view * target_vec
                target_flow = target_vec - base_sample
                predicted_signal = model(z_t, t, feat_g, feat_l)
                base_loss = F.mse_loss(predicted_signal, target_flow)
                pred_x0 = z_t + (1.0 - t_view) * predicted_signal

            # 反归一化得到物理空间 W
            pred_x0_raw = pred_x0 * target_std + target_mean
            target_x0_raw = target_vec * target_std + target_mean

            pred_W = pred_x0_raw.view(-1, 64, 32)
            target_W = target_x0_raw.view(-1, 64, 32)
            w_loss = F.mse_loss(pred_W, target_W)

            soh_loss = torch.tensor(0.0, device=device)
            if use_soh_loss:
                feat_w = build_X_from_cycles(w_cyc.reshape(-1, w_cyc.shape[-1]), encoder, device)
                feat_w = feat_w.view(w_cyc.shape[0], w_cyc.shape[1], -1).to(device)
                pred_W_batch = pred_W.repeat_interleave(feat_w.shape[1], dim=0)
                aligned = torch.bmm(feat_w.reshape(-1, feat_w.shape[-1]).unsqueeze(1), pred_W_batch).squeeze(1)
                pred_soh = soh_evaluator.head(aligned)
                soh_loss = F.smooth_l1_loss(pred_soh.view_as(target_soh), target_soh, beta=0.02)

            # 梯度强度采集（每10个epoch收集一次）
            if (epoch + 1) % 10 == 0 and use_soh_loss and soh_loss.requires_grad:
                grads_soh = torch.autograd.grad(
                    soh_loss, model.parameters(),
                    retain_graph=True,
                    create_graph=False
                )
                soh_grad_norm = torch.cat([g.reshape(-1) for g in grads_soh if g is not None]).norm(2).item()
            else:
                soh_grad_norm = None

            # 总损失
            loss = noise_w * base_loss + w_w * w_loss + soh_w * soh_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 若采集了 soh 梯度，则计算总梯度范数并累加
            if soh_grad_norm is not None:
                total_grad_vec = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
                total_grad_norm = total_grad_vec.norm(2).item()
                sum_soh_grad_norm += soh_grad_norm
                sum_total_grad_norm += total_grad_norm
                grad_count += 1

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        # 每个epoch输出一次平均梯度占比
        if grad_count > 0:
            avg_soh_norm = sum_soh_grad_norm / grad_count
            avg_total_norm = sum_total_grad_norm / grad_count
            avg_ratio = avg_soh_norm / (avg_total_norm + 1e-12)
            logging.info(f"Epoch {epoch+1} | Avg SOH grad norm: {avg_soh_norm:.6f} | "
                         f"Avg total grad norm: {avg_total_norm:.6f} | "
                         f"Avg ratio (soh/total): {avg_ratio:.4f}")

        # 更新 best_loss 与计数器
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info("Best model so far.")
            plateau_counter = 0
            early_stop_counter = 0
        else:
            plateau_counter += 1
            early_stop_counter += 1

        # 简化的阶段切换逻辑（仅两个状态）
        if not soh_phase_activated and plateau_counter >= 100:
            in_soh_phase = True
            soh_phase_activated = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = 2e-4
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=50
            )
            logging.info('Switched to SOH-dominant phase, LR reset to 2e-4')

        # 早停
        if early_stop_counter >= 300:
            logging.info('Early stopping triggered after 300 epochs without improvement.')
            break

        logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                     f"Phase: {'SOH' if in_soh_phase else 'Base'} | "
                     f"Weights: noise={noise_w:.2f} W={w_w:.2f} SOH={soh_w:.2f}")
# ==============================================================================
# 4. 测试与推理逻辑 (反归一化还原)
# ==============================================================================

def test_diffusion(model_path, stats_path, encoder, weights_dir, labeled_dir, device, local_num, global_num, result_path, diffusion_steps, denoiser_hidden_dim, denoiser_type, cond_input_dim=100, init_vec=None):
    def calc_metrics(pred, target):
        """计算 MSE, MAE, RMSE, MAPE"""
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        mse = np.mean((pred - target) ** 2)
        mae = np.mean(np.abs(pred - target))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((target - pred) / (np.abs(target) + 1e-8))) * 100
        return mse, mae, rmse, mape

    # 加载训练阶段保存的均值和标准差，用于反归一化
    stats = torch.load(stats_path, map_location=device)
    t_mean, t_std = stats['mean'].to(device), stats['std'].to(device)

    # 与 06 端到端评估保持一致：采样前先将 init_vec 标准化到训练空间
    init_vec_norm = None
    if init_vec is not None:
        init_vec_norm = ((init_vec.to(device).view(-1) - t_mean) / (t_std + 1e-8)).view(1, -1)

    diff_manager = DiffusionManager(T=diffusion_steps, device=device)
    flow_manager = FlowMatchingManager(T=diffusion_steps, device=device)
    denoiser_cls = WeightDenoiser if denoiser_type == "diff" else FlowMatchingDenoiser
    model = denoiser_cls(weight_dim=W_DIM, hidden_dim=denoiser_hidden_dim, cond_input_dim=cond_input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    test_subdirs = sorted([d for d in weights_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])

    for subdir in test_subdirs:
        weight_files = sorted(subdir.glob('*.pkl'))

        # 仅保留 W 指标
        metrics_W = {'mse': [], 'mae': [], 'rmse': [], 'mape': []}

        for f in tqdm(weight_files, desc=f"评估 {subdir.name}"):
            target_vec_raw, l_cyc, g_cyc = load_data_flattened(f, device, labeled_dir / subdir.name, local_num, global_num)

            feat_l = l_cyc.unsqueeze(0).to(device).float()
            feat_g = g_cyc.unsqueeze(0).to(device).float()

            if denoiser_type == "diff":
                gen_vec_norm = diff_manager.p_sample_loop(model, feat_g, feat_l, (1, W_DIM), init_vec=init_vec_norm)
            else:
                gen_vec_norm = flow_manager.sample(model, feat_g, feat_l, (1, W_DIM), init_vec=init_vec_norm)

            gen_vec = gen_vec_norm * t_std + t_mean
            pred_W = gen_vec.squeeze().view(64, 32)
            target_W = target_vec_raw.to(device).view(64, 32)

            mse_w, mae_w, rmse_w, mape_w = calc_metrics(pred_W.flatten(), target_W.flatten())
            metrics_W['mse'].append(mse_w)
            metrics_W['mae'].append(mae_w)
            metrics_W['rmse'].append(rmse_w)
            metrics_W['mape'].append(mape_w)

        if len(metrics_W['mse']) > 0:
            row = {
                'dataset': subdir.name,
                'mse_W': np.mean(metrics_W['mse']),
                'mae_W': np.mean(metrics_W['mae']),
                'rmse_W': np.mean(metrics_W['rmse']),
                'mape_W': np.mean(metrics_W['mape']),
            }
            results.append(row)
            logging.info(f"数据集 {subdir.name} | RMSE_W: {row['rmse_W']:.6f}")

    columns = ['dataset', 'mse_W', 'mae_W', 'rmse_W', 'mape_W']
    pd.DataFrame(results, columns=columns).to_csv(result_path, index=False)
    logging.info(f"结果已保存到: {result_path}")

# 5. 主程序执行入口
# ==============================================================================

if __name__ == "__main__":
    # 配置初始化
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="main")
    config = OmegaConf.create(config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=200)    
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--local_num_cycles", type=int, default=5)
    parser.add_argument("--global_num_cycles", type=int, default=20)
    parser.add_argument("--diffusion_steps", type=int, default=400)
    parser.add_argument("--denoiser_hidden_dim", type=int, default=512)
    parser.add_argument("--denoiser_type", type=str, default="diff", choices=["diff", "flow"])
    parser.add_argument("--use_soh_loss", action="store_true", default=True, help="Enable direct SOH supervision")
    parser.add_argument("--use_log_s_reg", action="store_true", default=True, help="Enable independent log_s regularization")
    parser.add_argument("--use_dynamic_weights", action="store_true", default=True, help="Enable epoch-based multi-task weight annealing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义文件路径
    labeled_dir = Path(config.path.labeled_dir)
    weights_dir = Path(config.path.weights_dir) / f'wsize_{args.window_size}_stride_{args.stride}'
    results_dir = Path(config.path.diffusion_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    diffusion_models_dir = Path(config.path.diffusion_models_dir)
    diffusion_models_dir.mkdir(parents=True, exist_ok=True)
    gma_models_dir = Path(config.path.gma_models_dir)
    if args.denoiser_type == "diff":
        model_save_path = diffusion_models_dir / f'WeightDiffusion_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.pkl'
        stats_path = diffusion_models_dir / f'diffusion_stats_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.pth'
    else:
        model_save_path = diffusion_models_dir / f'WeightFlowMatching_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.pkl'
        stats_path = diffusion_models_dir / f'flow_matching_stats_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.pth'
    init_abs_path = gma_models_dir / 'GMA_pretrained_abs.pth'
    csv_path = results_dir / f'test_{args.denoiser_type}_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.csv'
    
    # 预加载特征提取器
    encoder = load_frozen_encoder(gma_models_dir / 'GMA-NET.pkl', device)
    soh_evaluator = load_frozen_soh_evaluator(gma_models_dir / 'GMA-NET.pkl', device)
    init_vec = load_pretrained_abs_init(init_abs_path, device)

    # 第一阶段：训练数据预处理与 Z-Score 统计计算
    train_files = sorted((weights_dir / 'train').glob('*.pkl'))
    raw_cache = []
    all_targets = []
    
    for f in tqdm(train_files, desc="缓存训练数据"):
        target, l_cyc, g_cyc = load_data_flattened(f, device, labeled_dir/'train', args.local_num_cycles, args.global_num_cycles)
        target_soh = load_target_soh(f, labeled_dir/'train', args.window_size)
        w_cyc = load_window_cycles(f, labeled_dir/'train', args.window_size)
        # local/global 条件不再过冻结 encoder，直接使用原始循环序列
        f_l = l_cyc.float()
        f_g = g_cyc.float()
        all_targets.append(target)
        raw_cache.append((target, f_l, f_g, w_cyc, target_soh))
    
    # 统计 2048 维 W 向量在整个训练集上的均值和标准差
    all_targets_tensor = torch.stack(all_targets)
    t_mean = all_targets_tensor.mean(dim=0)
    t_std = all_targets_tensor.std(dim=0) + 1e-6
    torch.save({'mean': t_mean, 'std': t_std}, stats_path)

    # 推断 local/global 原始循环向量维度，供 denoiser 条件投影层使用
    if not raw_cache:
        raise RuntimeError('No training samples found when inferring cond_input_dim.')
    cond_input_dim = int(raw_cache[0][1].shape[-1])

    # 应用归一化到缓存数据中
    norm_cached_data = [((t - t_mean)/t_std, fl, fg, w_cyc, target_soh) for t, fl, fg, w_cyc, target_soh in raw_cache]
    train_loader = DataLoader(HyperCacheDataset(norm_cached_data), batch_size=64, shuffle=True)

    # 第二阶段：执行扩散模型去噪网络训练
    logging.info("--- 开始训练阶段 ---")
    train_diffusion(
        train_loader,
        model_save_path,
        device,
        diffusion_steps=args.diffusion_steps,
        denoiser_hidden_dim=args.denoiser_hidden_dim,
        denoiser_type=args.denoiser_type,
        window_size=args.window_size,
        target_mean=t_mean,
        target_std=t_std,
        init_vec=init_vec,
        encoder=encoder,
        soh_evaluator=soh_evaluator,
        cond_input_dim=cond_input_dim,
        use_soh_loss=args.use_soh_loss,
        use_log_s_reg=args.use_log_s_reg,
        use_dynamic_weights=args.use_dynamic_weights,
    )

    # 第三阶段：推理生成并评估效果
    logging.info("--- 开始测试评估阶段 ---")
    test_diffusion(
        model_save_path,
        stats_path,
        encoder,
        weights_dir,
        labeled_dir,
        device,
        args.local_num_cycles,
        args.global_num_cycles,
        csv_path,
        diffusion_steps=args.diffusion_steps,
        denoiser_hidden_dim=args.denoiser_hidden_dim,
        denoiser_type=args.denoiser_type,
        cond_input_dim=cond_input_dim,
        init_vec=init_vec
    )