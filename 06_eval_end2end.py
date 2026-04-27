import logging
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import argparse
import torch.nn.functional as F

from battery_data import BatteryData
from model import GMANet, WeightDenoiser, FlowMatchingDenoiser  # 确保导入扩散模型类


def infer_cycle_input_dim(weights_dir, labeled_dir):
    """Infer raw cycle feature length from the first available test sample."""
    test_subdirs = sorted([d for d in weights_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])
    for subdir in test_subdirs:
        weight_files = sorted(subdir.glob('*.pkl'))
        if not weight_files:
            continue
        f = weight_files[0]
        parts = f.stem.split("_")
        battery_name = "_".join(parts[:-1])
        battery_path = labeled_dir / subdir.name / f"{battery_name}.pkl"
        if not battery_path.exists():
            continue
        battery = BatteryData.load(str(battery_path))
        if len(battery.cycle_data) == 0:
            continue
        return int(np.asarray(battery.cycle_data[0].labeled_Qc, dtype=np.float32).shape[-1])
    raise RuntimeError('Unable to infer cycle input dimension from test data.')


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

# ==========================================
# 1. 扩散采样器 (用于测试时的 Reverse Process)
# ==========================================

class DiffusionSampler:
    def __init__(self, T=20, device='cuda'):
        self.T = T
        self.device = device
        # 需与训练时的调度完全一致
        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
    @torch.no_grad()
    def sample(self, model, feat_g, feat_l, init_vec=None):
        """
        输入: 
            model: 训练好的 WeightDenoiser
            feat_g, feat_l: 提取好的全局和局部特征 [1, Seq, 64]
        返回: 
            归一化空间的展平权重向量 [1, 385]
        """
        model.eval()
        # 逆扩散起点：优先使用预训练 GMA 的 A/B/S 原型向量，否则回退随机噪声
        if init_vec is not None:
            cur_x = init_vec.to(self.device).view(1, -1)
        else:
            cur_x = torch.randn((1, 385), device=self.device)
        
        for i in reversed(range(self.T)):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            # 预测噪声
            predicted_noise = model(cur_x, t, feat_g, feat_l)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            noise = torch.randn_like(cur_x) if i > 0 else 0
            
            # DDPM 迭代公式
            cur_x = (1 / torch.sqrt(alpha_t)) * (
                cur_x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

            # 与训练采样保持一致：防止离群值在反归一化后放大
            cur_x = torch.clamp(cur_x, -5.0, 5.0)

        return cur_x


class FlowMatchingSampler:
    def __init__(self, T=20, device='cuda'):
        self.T = T
        self.device = device

    @torch.no_grad()
    def sample(self, model, feat_g, feat_l, init_vec=None):
        model.eval()
        if init_vec is None:
            cur_x = torch.randn((1, 385), device=self.device)
        else:
            cur_x = init_vec.view(1, -1)
        dt = 1.0 / max(self.T, 1)

        for i in range(self.T):
            t_value = i / max(self.T, 1)
            t = torch.full((1,), t_value, device=self.device, dtype=torch.float32)
            velocity = model(cur_x, t, feat_g, feat_l)
            cur_x = cur_x + dt * velocity
            cur_x = torch.clamp(cur_x, -5.0, 5.0)

        return cur_x

# ==========================================
# 2. 核心端到端推理逻辑
# ==========================================

def run_diffusion_end2end_evaluation(base_model, denoiser, sampler, stats, weights_dir, labeled_dir, device, args, result_path, init_vec=None):
    """
    针对扩散模型优化的端到端评估
    """
    detailed_results = []
    
    # 提取 Z-Score 统计量
    t_mean = stats['mean'].to(device)
    t_std = stats['std'].to(device)

    # 扩散模型在标准化空间训练，初始化向量需先标准化
    init_vec_norm = None
    if init_vec is not None:
        init_vec_norm = ((init_vec.to(device).view(-1) - t_mean) / (t_std + 1e-8)).view(1, -1)

    test_subdirs = sorted([d for d in weights_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])
    
    if not test_subdirs:
        logging.error("未找到测试子目录。")
        return

    for subdir in test_subdirs:
        dataset_name = subdir.name
        raw_data_dir = labeled_dir / dataset_name
        weight_files = sorted(subdir.glob('*.pkl'))
        
        logging.info(f"正在深度评估扩散模型: {dataset_name}")
        
        for f in tqdm(weight_files, desc=f"Eval {dataset_name}"):
            # --- 1. 元数据解析 ---
            parts = f.stem.split("_")
            start_idx = int(parts[-1])
            battery_name = "_".join(parts[:-1])
            battery_path = raw_data_dir / f"{battery_name}.pkl"
            
            if not battery_path.exists(): continue
            battery = BatteryData.load(str(battery_path))
            
            # --- 2. 准备信号 ---
            g_data = battery.cycle_data[0 : args.global_num_cycles]
            g_cyc = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in g_data])).to(device)
            
            l_data = battery.cycle_data[start_idx : start_idx + args.local_num_cycles]
            l_cyc = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in l_data])).to(device)
            
            w_data = battery.cycle_data[start_idx : start_idx + args.window_size]
            w_cyc = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in w_data])).to(device)
            true_soh = np.array([c.labeled_soh for c in w_data])

            # --- 3. 扩散采样推理 ---
            with torch.no_grad():
                # A. 提取指纹特征
                feat_g = g_cyc.unsqueeze(0).float()
                feat_l = l_cyc.unsqueeze(0).float()
                
                # B. 采样生成标准化向量 (1, 385)
                gen_vec_norm = sampler.sample(denoiser, feat_g, feat_l, init_vec=init_vec_norm)
                
                # C. 反标准化 (回到真实物理量级)
                gen_vec = gen_vec_norm * t_std + t_mean
                
                # D. 拆解参数: A(4*64=256), B(4*32=128), log_s(1)
                # A: 64*4 = 256, B: 4*32 = 128
                p_A = gen_vec[:, 0:256].view(1, 64, 4)  # 对应 apply_lora 里的形状
                p_B = gen_vec[:, 256:384].view(1, 4, 32)
                p_log_s = torch.clamp(gen_vec[:, 384:385], min=-args.log_s_clip, max=args.log_s_clip)
                
                # E. 注入 LoRA 权重预测 SOH
                feat_w = base_model.encoder(w_cyc)
                p_A_batch = p_A.repeat(feat_w.size(0), 1, 1)
                p_B_batch = p_B.repeat(feat_w.size(0), 1, 1)
                p_log_s_batch = p_log_s.repeat(feat_w.size(0), 1)
                
                pred_soh = base_model.apply_lora(feat_w, p_A_batch, p_B_batch, p_log_s_batch)
                pred_soh = pred_soh.cpu().numpy().flatten()

            # --- 4. 误差计算 ---
            pred_soh_raw = pred_soh
            pred_soh_clip = np.clip(pred_soh_raw, 0.0, 1.0)

            mse_raw = np.mean((pred_soh_raw - true_soh) ** 2)
            mae_raw = np.mean(np.abs(pred_soh_raw - true_soh))
            rmse_raw = np.sqrt(mse_raw)
            # MAPE: 避免除以0，仅在 true_soh > 0 时计算
            mask = true_soh > 0
            mape_raw = np.mean(np.abs((true_soh[mask] - pred_soh_raw[mask]) / true_soh[mask])) if np.any(mask) else 0.0

            mse = np.mean((pred_soh_clip - true_soh) ** 2)
            mae = np.mean(np.abs(pred_soh_clip - true_soh))
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((true_soh[mask] - pred_soh_clip[mask]) / true_soh[mask])) if np.any(mask) else 0.0

            detailed_results.append({
                'dataset': dataset_name,
                'battery_name': battery_name,
                'window_start': start_idx,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'mse_raw': mse_raw,
                'mae_raw': mae_raw,
                'rmse_raw': rmse_raw,
                'mape_raw': mape_raw
            })

    df = pd.DataFrame(detailed_results)
    df.to_csv(result_path, index=False)
    
    if df.empty:
        logging.warning('No evaluation records were generated.')
        return

    summary_clip = df.groupby('dataset')[['mae', 'rmse', 'mape']].mean()
    summary_raw = df.groupby('dataset')[['mae_raw', 'rmse_raw', 'mape_raw']].mean()
    logging.info(f"\n=== 扩散模型评估摘要 (clipped [0,1]) ===\n{summary_clip}")
    logging.info(f"\n=== 扩散模型评估摘要 (raw) ===\n{summary_raw}")
    logging.info(f"报告已保存: {result_path}")

# ==========================================
# 3. Main
# ==========================================

if __name__ == "__main__":
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="main")
    config = OmegaConf.create(config)

    parser = argparse.ArgumentParser(description="Weight Diffusion End-to-End Evaluation")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--local_num_cycles", type=int, default=5)
    parser.add_argument("--global_num_cycles", type=int, default=20)
    parser.add_argument("--diffusion_steps", type=int, default=400)
    parser.add_argument("--denoiser_hidden_dim", type=int, default=512)
    parser.add_argument("--denoiser_type", type=str, default="diff", choices=["diff", "flow"])
    parser.add_argument("--log_s_clip", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    results_dir = Path(config.path.diffusion_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    diffusion_models_dir = Path(config.path.diffusion_models_dir)
    gma_models_dir = Path(config.path.gma_models_dir)
    
    labeled_dir = Path(config.path.labeled_dir)
    weights_dir = Path(config.path.weights_dir) / f'wsize_{args.window_size}_stride_{args.stride}'
    
    # 模型与统计量路径
    base_model_path = gma_models_dir / 'GMA-NET.pkl'
    if args.denoiser_type == "diff":
        diffusion_model_path = diffusion_models_dir / f'WeightDiffusion_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.pkl'
        stats_path = diffusion_models_dir / f'diffusion_stats_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.pth'
    else:
        diffusion_model_path = diffusion_models_dir / f'WeightFlowMatching_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.pkl'
        stats_path = diffusion_models_dir / f'flow_matching_stats_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.pth'
    init_abs_path = gma_models_dir / 'GMA_pretrained_abs.pth'
    
    result_path = results_dir / f'end2end_{args.denoiser_type}_local{args.local_num_cycles}_global{args.global_num_cycles}_steps{args.diffusion_steps}_dim{args.denoiser_hidden_dim}_wsize{args.window_size}_stride{args.stride}.csv'

    # --- 加载模型 ---
    # 1. Base Model (GMANet)
    base_model = GMANet(rank=4).to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device), strict=False)
    
    # 2. Denoiser (Diffusion or Flow Matching)
    denoiser_cls = WeightDenoiser if args.denoiser_type == "diff" else FlowMatchingDenoiser
    cond_input_dim = infer_cycle_input_dim(weights_dir, labeled_dir)
    denoiser = denoiser_cls(weight_dim=385, hidden_dim=args.denoiser_hidden_dim, cond_input_dim=cond_input_dim).to(device)
    denoiser.load_state_dict(torch.load(diffusion_model_path, map_location=device))
    
    # 3. Z-Score Stats
    if not stats_path.exists():
        raise FileNotFoundError(f"未找到统计量文件 {stats_path}，请先运行训练脚本生成。")
    stats = torch.load(stats_path, map_location=device)
    init_vec = load_pretrained_abs_init(init_abs_path, device)

    # 4. Sampler
    sampler = DiffusionSampler(T=args.diffusion_steps, device=device) if args.denoiser_type == "diff" else FlowMatchingSampler(T=args.diffusion_steps, device=device)

    base_model.eval()
    denoiser.eval()

    logging.info("--- 开始扩散模型端到端评估 ---")
    run_diffusion_end2end_evaluation(
        base_model, denoiser, sampler, stats,
        weights_dir, labeled_dir, device, args, result_path,
        init_vec=init_vec
    )