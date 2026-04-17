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
from model import GMANet, WeightDenoiser  # 确保导入扩散模型类

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
    def sample(self, model, feat_g, feat_l):
        """
        输入: 
            model: 训练好的 WeightDenoiser
            feat_g, feat_l: 提取好的全局和局部特征 [1, Seq, 64]
        返回: 
            归一化空间的展平权重向量 [1, 385]
        """
        model.eval()
        # 从纯噪声开始
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
            
        return cur_x

# ==========================================
# 2. 核心端到端推理逻辑
# ==========================================

def run_diffusion_end2end_evaluation(base_model, denoiser, sampler, stats, weights_dir, labeled_dir, device, args, result_path):
    """
    针对扩散模型优化的端到端评估
    """
    detailed_results = []
    
    # 提取 Z-Score 统计量
    t_mean = stats['mean'].to(device)
    t_std = stats['std'].to(device)

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
                feat_g = base_model.encoder(g_cyc).unsqueeze(0)
                feat_l = base_model.encoder(l_cyc).unsqueeze(0)
                
                # B. 采样生成标准化向量 (1, 385)
                gen_vec_norm = sampler.sample(denoiser, feat_g, feat_l)
                
                # C. 反标准化 (回到真实物理量级)
                gen_vec = gen_vec_norm * t_std + t_mean
                
                # D. 拆解参数: A(4*64=256), B(4*32=128), log_s(1)
                # A: 64*4 = 256, B: 4*32 = 128
                p_A = gen_vec[:, 0:256].view(1, 64, 4)  # 对应 apply_lora 里的形状
                p_B = gen_vec[:, 256:384].view(1, 4, 32)
                p_log_s = gen_vec[:, 384:385]
                
                # E. 注入 LoRA 权重预测 SOH
                feat_w = base_model.encoder(w_cyc)
                p_A_batch = p_A.repeat(feat_w.size(0), 1, 1)
                p_B_batch = p_B.repeat(feat_w.size(0), 1, 1)
                p_log_s_batch = p_log_s.repeat(feat_w.size(0), 1)
                
                pred_soh = base_model.apply_lora(feat_w, p_A_batch, p_B_batch, p_log_s_batch)
                pred_soh = pred_soh.cpu().numpy().flatten()

            # --- 4. 误差计算 ---
            mse = np.mean((pred_soh - true_soh) ** 2)
            mae = np.mean(np.abs(pred_soh - true_soh))
            rmse = np.sqrt(mse)

            detailed_results.append({
                'dataset': dataset_name,
                'battery_name': battery_name,
                'window_start': start_idx,
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            })

    df = pd.DataFrame(detailed_results)
    df.to_csv(result_path, index=False)
    
    summary = df.groupby('dataset')[['mae', 'rmse']].mean()
    logging.info(f"\n=== 扩散模型评估摘要 ===\n{summary}")
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
    parser.add_argument("--local_num_cycles", type=int, default=5)
    parser.add_argument("--global_num_cycles", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    res_dir = Path(config.path.results_dir)
    e2e_dir = res_dir / "end2end_diffusion_results"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    
    labeled_dir = Path(config.path.labeled_dir)
    weights_dir = Path(config.path.weights_dir) / f'wsize_{args.window_size}_stride_50' # 假设 stride 50
    
    # 模型与统计量路径
    base_model_path = res_dir / 'GMA-NET.pkl'
    diffusion_model_path = res_dir / 'WeightDiffusion_Denoiser.pkl'
    stats_path = res_dir / 'diffusion_stats.pth'
    
    result_path = e2e_dir / f'Diffusion_E2E_wsize{args.window_size}.csv'

    # --- 加载模型 ---
    # 1. Base Model (GMANet)
    base_model = GMANet(rank=4).to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device), strict=False)
    
    # 2. Diffusion Denoiser
    denoiser = WeightDenoiser(weight_dim=385, hidden_dim=256).to(device)
    denoiser.load_state_dict(torch.load(diffusion_model_path, map_location=device))
    
    # 3. Z-Score Stats
    if not stats_path.exists():
        raise FileNotFoundError(f"未找到统计量文件 {stats_path}，请先运行训练脚本生成。")
    stats = torch.load(stats_path, map_location=device)

    # 4. Sampler
    sampler = DiffusionSampler(T=1000, device=device)

    base_model.eval()
    denoiser.eval()

    logging.info("--- 开始扩散模型端到端评估 ---")
    run_diffusion_end2end_evaluation(
        base_model, denoiser, sampler, stats, 
        weights_dir, labeled_dir, device, args, result_path
    )