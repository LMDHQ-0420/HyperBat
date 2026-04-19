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
from model import GMANet, GMANetPreTrain, WeightDenoiser 

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
    1. 将 LoRA 的 A 和 B 矩阵分解为“方向”和“能量(log_s)”
    2. 将其拼接成一个 385 维的平坦向量，作为扩散模型生成的目标
    """
    weights = torch.load(weight_path, map_location=device)
    p_A, p_B = weights["param_A"].to(device).float(), weights["param_B"].to(device).float()

    # 如果上游已显式保存 log_s，则优先使用该值；否则由 A/B 范数恢复
    A_norm = p_A / (torch.norm(p_A, p="fro") + eps)
    B_norm = p_B / (torch.norm(p_B, p="fro") + eps)
    if "param_log_s" in weights:
        log_s = weights["param_log_s"].to(device).float().view(1)
    else:
        norm_val = torch.norm(p_A, p="fro") * torch.norm(p_B, p="fro")
        log_s = torch.log(norm_val + eps).view(1)

    # 目标向量组成：A(16x16=256) + B(16x8=128) + log_s(1) = 385 维
    target_vec = torch.cat([A_norm.flatten(), B_norm.flatten(), log_s])

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


def load_pretrained_abs_init(init_abs_path, device):
    """加载 GMA 预训练导出的 A/B/S 原型并转换为 385 维初始化向量。"""
    if not init_abs_path.exists():
        logging.warning(f'ABS init file not found: {init_abs_path}. Fallback to random noise.')
        return None

    data = torch.load(init_abs_path, map_location=device)
    if 'init_vec' in data:
        init_vec = data['init_vec'].float().view(-1)
    elif 'param_A' in data and 'param_B' in data and 'param_log_s' in data:
        init_vec = torch.cat([
            data['param_A'].float().view(-1),
            data['param_B'].float().view(-1),
            data['param_log_s'].float().view(-1),
        ])
    else:
        logging.warning(f'Invalid ABS init format: {init_abs_path}. Fallback to random noise.')
        return None

    if init_vec.numel() != 385:
        logging.warning(f'ABS init size mismatch ({init_vec.numel()}). Expected 385. Fallback to random noise.')
        return None

    return init_vec.to(device)

# ==============================================================================
# 3. 核心训练逻辑
# ==============================================================================

def train_diffusion(
    train_loader,
    model_save_path,
    device,
    diffusion_steps,
    denoiser_hidden_dim,
    window_size,
    target_mean,
    target_std,
    soh_evaluator,
    epochs=100000,
    use_soh_loss=True,
    use_log_s_reg=True,
    use_dynamic_weights=True,
):
    # 扩散步数与网络宽度需和评估保持一致
    diff_manager = DiffusionManager(T=diffusion_steps, device=device)

    # 定义去噪网络：输入 (噪声向量, 时间步, 全局特征, 局部特征)
    model = WeightDenoiser(weight_dim=385, hidden_dim=denoiser_hidden_dim).to(device)
    
    # 移动目标统计值到设备
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    
    # 动态学习率调度：监控 Loss，若多次不降则降低 LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 300 
    
    logging.info("--- 开始扩散模型增强训练 ---")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for target_vec, feat_l, feat_g, target_soh in train_loader:
            target_vec = target_vec.to(device)
            feat_l = feat_l.to(device)
            feat_g = feat_g.to(device)
            target_soh = target_soh.to(device)
            
            # 随机采样时间步 t
            t = torch.randint(0, diff_manager.T, (target_vec.shape[0],), device=device).long()
            
            # 生成标准高斯噪声
            noise = torch.randn_like(target_vec)
            
            # 前向扩散过程，获得加噪后的样本 z_t
            z_t = diff_manager.q_sample(target_vec, t, noise)
            
            optimizer.zero_grad()
            
            # 模型预测噪声
            predicted_noise = model(z_t, t, feat_g, feat_l)
            
            # 1) 噪声预测损失（基础 DDPM 目标）
            noise_loss_map = F.mse_loss(predicted_noise, noise, reduction='none')
            noise_loss_map[:, -1] *= 20.0
            noise_loss = noise_loss_map.mean()

            # 2) 从 epsilon 反推 x0，增强与真实参数向量的一致性
            sqrt_ab_t = torch.sqrt(diff_manager.alphas_cumprod[t]).view(-1, 1)
            sqrt_one_minus_ab_t = torch.sqrt(1 - diff_manager.alphas_cumprod[t]).view(-1, 1)
            pred_x0 = (z_t - sqrt_one_minus_ab_t * predicted_noise) / (sqrt_ab_t + 1e-8)

            vec_loss = F.mse_loss(pred_x0[:, :384], target_vec[:, :384])
            pred_log_s = pred_x0[:, 384:385]
            target_log_s = target_vec[:, 384:385]
            log_s_fit_loss = F.smooth_l1_loss(pred_log_s, target_log_s, beta=0.1)
            if use_log_s_reg:
                log_s_mean = target_log_s.mean().detach()
                log_s_std = target_log_s.std().detach().clamp_min(1e-8)
                log_s_reg_loss = torch.mean(((pred_log_s - log_s_mean) / log_s_std) ** 2)
                log_s_loss = log_s_fit_loss + 0.3 * log_s_reg_loss
            else:
                log_s_loss = log_s_fit_loss

            # 3) 在下游实际使用的 W 空间做一致性约束，减小“向量 MSE 好但 SOH 差”
            pred_x0_raw = pred_x0 * target_std + target_mean
            target_x0_raw = target_vec * target_std + target_mean

            p_A = pred_x0_raw[:, 0:256].view(-1, 64, 4)
            p_B = pred_x0_raw[:, 256:384].view(-1, 4, 32)
            p_log_s = torch.clamp(pred_x0_raw[:, 384:385], min=-5.0, max=5.0)

            t_A = target_x0_raw[:, 0:256].view(-1, 64, 4)
            t_B = target_x0_raw[:, 256:384].view(-1, 4, 32)
            t_log_s = torch.clamp(target_x0_raw[:, 384:385], min=-5.0, max=5.0)

            pred_W = torch.exp(p_log_s).view(-1, 1, 1) * torch.bmm(p_A, p_B)
            target_W = torch.exp(t_log_s).view(-1, 1, 1) * torch.bmm(t_A, t_B)
            w_loss = F.mse_loss(pred_W, target_W)

            soh_loss = torch.tensor(0.0, device=device)
            if use_soh_loss:
                pooled_feat = 0.5 * (feat_l.mean(dim=1) + feat_g.mean(dim=1))
                pred_soh = soh_evaluator.apply_lora(pooled_feat, p_A, p_B, p_log_s)
                pred_soh_window = pred_soh.expand(-1, target_soh.size(1))
                soh_loss = F.smooth_l1_loss(pred_soh_window, target_soh, beta=0.02)

            # 4) 多任务损失权重动态调整
            if use_dynamic_weights:
                progress = min((epoch + 1) / max(epochs, 1), 1.0)
                noise_w = 1.0 * (1 - 0.3 * progress)
                vec_w = 0.2 * (1 + 1.0 * progress)
                log_s_w = 0.5 * (1 + 0.5 * progress)
                w_w = 0.2 * (1 + 1.5 * progress)
                soh_w = 0.8 * (0.5 + 1.0 * progress)
            else:
                noise_w, vec_w, log_s_w, w_w, soh_w = 1.0, 0.2, 0.5, 0.2, 0.8

            loss = noise_w * noise_loss + vec_w * vec_loss + log_s_w * log_s_loss + w_w * w_loss + soh_w * soh_loss
            
            loss.backward()
            
            # 梯度裁剪，防止因训练初期预测不准导致的梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停与模型保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            logging.info(f"触发早停条件，停止于第 {epoch+1} 轮。")
            break

# ==============================================================================
# 4. 测试与推理逻辑 (反归一化还原)
# ==============================================================================

def test_diffusion(model_path, stats_path, encoder, weights_dir, labeled_dir, device, local_num, global_num, result_path, diffusion_steps, denoiser_hidden_dim, init_vec=None):
    # 加载训练阶段保存的均值和标准差，用于反归一化
    stats = torch.load(stats_path, map_location=device)
    t_mean, t_std = stats['mean'].to(device), stats['std'].to(device)
    
    diff_manager = DiffusionManager(T=diffusion_steps, device=device)
    model = WeightDenoiser(weight_dim=385, hidden_dim=denoiser_hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    test_subdirs = sorted([d for d in weights_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])
    
    for subdir in test_subdirs:
        weight_files = sorted(subdir.glob('*.pkl'))
        mse_accum, count = 0, 0
        for f in tqdm(weight_files, desc=f"评估 {subdir.name}"):
            # 得到物理量级的 target_vec
            target_vec_raw, l_cyc, g_cyc = load_data_flattened(f, device, labeled_dir/subdir.name, local_num, global_num)
            
            # 提取特征
            feat_l = build_X_from_cycles(l_cyc, encoder, device).unsqueeze(0).to(device)
            feat_g = build_X_from_cycles(g_cyc, encoder, device).unsqueeze(0).to(device)
            
            # 进行逆向去噪采样（推理生成的也是标准归一化空间的向量）
            gen_vec_norm = diff_manager.p_sample_loop(model, feat_g, feat_l, (1, 385), init_vec=init_vec)
            
            # 关键：反归一化还原回物理量级空间
            gen_vec = gen_vec_norm * t_std + t_mean 
            
            # 计算在物理空间下的 MSE
            mse = F.mse_loss(gen_vec.squeeze(), target_vec_raw.to(device)).item()
            mse_accum += mse
            count += 1
        
        if count > 0:
            results.append([subdir.name, mse_accum/count])
            logging.info(f"数据集 {subdir.name} | 平均 MSE: {mse_accum/count:.6f}")

    pd.DataFrame(results, columns=['dataset', 'total_mse']).to_csv(result_path, index=False)

# ==============================================================================
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
    parser.add_argument("--use_soh_loss", action="store_true", default=True, help="Enable direct SOH supervision")
    parser.add_argument("--use_log_s_reg", action="store_true", default=True, help="Enable independent log_s regularization")
    parser.add_argument("--use_dynamic_weights", action="store_true", default=True, help="Enable epoch-based multi-task weight annealing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义文件路径
    labeled_dir = Path(config.path.labeled_dir)
    weights_dir = Path(config.path.weights_dir) / f'wsize_{args.window_size}_stride_{args.stride}'
    res_dir = Path(config.path.results_dir)
    csv_dir = res_dir / "diffusion_results"
    csv_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = res_dir / f'WeightDiffusion_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.pkl'
    stats_path = res_dir / 'diffusion_stats.pth'
    init_abs_path = res_dir / 'GMA_pretrained_abs.pth'
    csv_path = csv_dir / f'test_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.csv'
    
    # 预加载特征提取器
    encoder = load_frozen_encoder(res_dir / 'GMA-NET.pkl', device)
    soh_evaluator = load_frozen_soh_evaluator(res_dir / 'GMA-NET.pkl', device)
    init_vec = load_pretrained_abs_init(init_abs_path, device)

    # 第一阶段：训练数据预处理与 Z-Score 统计计算
    train_files = sorted((weights_dir / 'train').glob('*.pkl'))
    raw_cache = []
    all_targets = []
    
    for f in tqdm(train_files, desc="缓存训练数据"):
        target, l_cyc, g_cyc = load_data_flattened(f, device, labeled_dir/'train', args.local_num_cycles, args.global_num_cycles)
        target_soh = load_target_soh(f, labeled_dir/'train', args.window_size)
        f_l = build_X_from_cycles(l_cyc, encoder, device)
        f_g = build_X_from_cycles(g_cyc, encoder, device)
        all_targets.append(target)
        raw_cache.append((target, f_l, f_g, target_soh))
    
    # 统计 385 维向量在整个训练集上的均值和标准差（必须在训练前做，扩散模型要求输入均值为 0）
    all_targets_tensor = torch.stack(all_targets)
    t_mean = all_targets_tensor.mean(dim=0)
    t_std = all_targets_tensor.std(dim=0) + 1e-6
    torch.save({'mean': t_mean, 'std': t_std}, stats_path)

    # 应用归一化到缓存数据中
    norm_cached_data = [((t - t_mean)/t_std, fl, fg, target_soh) for t, fl, fg, target_soh in raw_cache]
    train_loader = DataLoader(HyperCacheDataset(norm_cached_data), batch_size=64, shuffle=True)

    # 第二阶段：执行扩散模型去噪网络训练
    logging.info("--- 开始训练阶段 ---")
    train_diffusion(
        train_loader,
        model_save_path,
        device,
        diffusion_steps=args.diffusion_steps,
        denoiser_hidden_dim=args.denoiser_hidden_dim,
        window_size=args.window_size,
        target_mean=t_mean,
        target_std=t_std,
        soh_evaluator=soh_evaluator,
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
        init_vec=init_vec
    )