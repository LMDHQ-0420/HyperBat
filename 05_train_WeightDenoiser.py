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

from battery_data import BatteryData
from model import GMANetPreTrain, WeightDenoiser 

# ==========================================
# 1. 扩散调度类 (Simple DDPM Scheduler)
# ==========================================

class DiffusionManager:
    def __init__(self, T=1000, device='cuda'):
        self.T = T
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample_loop(self, model, global_feat, local_feat, shape):
        model.eval()
        cur_x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            predicted_noise = model(cur_x, t, global_feat, local_feat)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            noise = torch.randn_like(cur_x) if i > 0 else 0
            cur_x = (1 / torch.sqrt(alpha_t)) * (
                cur_x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise
        return cur_x

class HyperCacheDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

# ==========================================
# 2. 基础工具函数
# ==========================================

def load_frozen_encoder(pretrained_path, device):
    model = GMANetPreTrain().to(device)
    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict)
    encoder = model.encoder
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    return encoder

def build_X_from_cycles(cycles, frozen_encoder, device):
    cycles_tensor = torch.from_numpy(cycles).float() if isinstance(cycles, np.ndarray) else cycles.float()
    frozen_encoder.eval()
    with torch.no_grad():
        feat = frozen_encoder(cycles_tensor.to(device))
    return feat.cpu()

def load_data_flattened(weight_path, device, train_dir, local_num, global_num, eps=1e-8):
    weights = torch.load(weight_path, map_location=device)
    p_A, p_B = weights["param_A"].to(device).float(), weights["param_B"].to(device).float()

    # 局部归一化 (保持 A, B 的量级稳定)
    norm_val = torch.norm(p_A, p="fro") * torch.norm(p_B, p="fro")
    A_norm = p_A / (torch.norm(p_A, p="fro") + eps)
    B_norm = p_B / (torch.norm(p_B, p="fro") + eps)
    log_s = torch.log(norm_val + eps).view(1)

    target_vec = torch.cat([A_norm.flatten(), B_norm.flatten(), log_s])

    parts = weight_path.stem.split("_")
    start_idx = int(parts[-1])
    battery_name = "_".join(parts[:-1])
    battery_path = train_dir / f"{battery_name}.pkl"
    battery = BatteryData.load(str(battery_path))

    g_cyc = np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in battery.cycle_data[0:global_num]])
    l_cyc = np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in battery.cycle_data[start_idx:start_idx+local_num]])

    return target_vec.cpu(), torch.tensor(l_cyc), torch.tensor(g_cyc)

# ==========================================
# 3. 训练与测试逻辑
# ==========================================

def train_diffusion(train_loader, model_save_path, device, epochs=1000):
    diff_manager = DiffusionManager(T=1000, device=device)
    model = WeightDenoiser(weight_dim=385, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for target_vec, feat_l, feat_g in train_loader:
            target_vec, feat_l, feat_g = target_vec.to(device), feat_l.to(device), feat_g.to(device)
            
            t = torch.randint(0, diff_manager.T, (target_vec.shape[0],), device=device).long()
            noise = torch.randn_like(target_vec)
            z_t = diff_manager.q_sample(target_vec, t, noise)
            
            optimizer.zero_grad()
            predicted_noise = model(z_t, t, feat_g, feat_l)
            
            # 损失计算：对最后一维 log_s 加权，因为它是 SOH 的关键缩放因子
            base_loss = F.mse_loss(predicted_noise, noise, reduction='none')
            base_loss[:, -1] *= 5.0 
            loss = base_loss.mean()
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)

def test_diffusion(model_path, stats_path, encoder, weights_dir, labeled_dir, device, local_num, global_num, result_path):
    stats = torch.load(stats_path, map_location=device)
    t_mean, t_std = stats['mean'].to(device), stats['std'].to(device)
    
    diff_manager = DiffusionManager(T=1000, device=device)
    model = WeightDenoiser(weight_dim=385, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    test_subdirs = sorted([d for d in weights_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])
    
    for subdir in test_subdirs:
        weight_files = sorted(subdir.glob('*.pkl'))
        mse_accum, count = 0, 0
        for f in tqdm(weight_files, desc=f"Eval {subdir.name}"):
            target_vec_raw, l_cyc, g_cyc = load_data_flattened(f, device, labeled_dir/subdir.name, local_num, global_num)
            feat_l = build_X_from_cycles(l_cyc, encoder, device).unsqueeze(0).to(device)
            feat_g = build_X_from_cycles(g_cyc, encoder, device).unsqueeze(0).to(device)
            
            # 采样并反归一化
            gen_vec_norm = diff_manager.p_sample_loop(model, feat_g, feat_l, (1, 385))
            gen_vec = gen_vec_norm * t_std + t_mean # 回到原始物理量级
            
            mse = F.mse_loss(gen_vec.squeeze(), target_vec_raw.to(device)).item()
            mse_accum += mse
            count += 1
        
        if count > 0:
            results.append([subdir.name, mse_accum/count])
            logging.info(f"Result {subdir.name} | MSE: {mse_accum/count:.6f}")

    pd.DataFrame(results, columns=['dataset', 'total_mse']).to_csv(result_path, index=False)

# ==========================================
# 4. Main
# ==========================================

if __name__ == "__main__":
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="main")
    config = OmegaConf.create(config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=200)    
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--local_num_cycles", type=int, default=5)
    parser.add_argument("--global_num_cycles", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径对齐
    labeled_dir = Path(config.path.labeled_dir)
    weights_dir = Path(config.path.weights_dir) / f'wsize_{args.window_size}_stride_{args.stride}'
    res_dir = Path(config.path.results_dir)
    model_save_path = res_dir / 'WeightDiffusion_Denoiser.pkl'
    stats_path = res_dir / 'diffusion_stats.pth'
    
    encoder = load_frozen_encoder(res_dir / 'GMA-NET.pkl', device)

    # 1. 数据预处理与 Z-Score 统计
    train_files = sorted((weights_dir / 'train').glob('*.pkl'))
    raw_cache = []
    all_targets = []
    for f in tqdm(train_files, desc="Processing Train Data"):
        target, l_cyc, g_cyc = load_data_flattened(f, device, labeled_dir/'train', args.local_num_cycles, args.global_num_cycles)
        f_l = build_X_from_cycles(l_cyc, encoder, device)
        f_g = build_X_from_cycles(g_cyc, encoder, device)
        all_targets.append(target)
        raw_cache.append((target, f_l, f_g))
    
    # 计算并保存统计量
    all_targets_tensor = torch.stack(all_targets)
    t_mean = all_targets_tensor.mean(dim=0)
    t_std = all_targets_tensor.std(dim=0) + 1e-6
    torch.save({'mean': t_mean, 'std': t_std}, stats_path)
    logging.info(f"Stats saved. Mean: {t_mean.mean():.4f}, Std: {t_std.mean():.4f}")

    # 应用归一化
    norm_cached_data = [((t - t_mean)/t_std, fl, fg) for t, fl, fg in raw_cache]
    train_loader = DataLoader(HyperCacheDataset(norm_cached_data), batch_size=64, shuffle=True)

    # 2. 训练
    logging.info("--- Starting Diffusion Training (Normalized) ---")
    train_diffusion(train_loader, model_save_path, device)

    # 3. 测试
    logging.info("--- Starting Diffusion Testing (De-normalized) ---")
    test_diffusion(model_save_path, stats_path, encoder, weights_dir, labeled_dir, device, 
                   args.local_num_cycles, args.global_num_cycles, res_dir/'diffusion_test_results.csv')