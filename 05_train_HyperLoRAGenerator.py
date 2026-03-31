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
from torch.utils.data import DataLoader, Dataset, Subset
import argparse

from battery_data import BatteryData
from model import GMANetPreTrain, HyperLoRAGenerator

# ==========================================
# 1. 基础工具函数
# ==========================================

def load_frozen_encoder(pretrained_path, device):
    """加载并冻结预训练的 Encoder"""
    model = GMANetPreTrain().to(device)
    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict)
    encoder = model.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder

def build_X_from_cycles(cycles, frozen_encoder, device):
    """特征提取：信号 [N, 400] -> 特征 [N, 64]"""
    if isinstance(cycles, np.ndarray):
        cycles_tensor = torch.from_numpy(cycles).float()
    else:
        cycles_tensor = cycles.float()

    frozen_encoder.eval()
    with torch.no_grad():
        # Batch 处理以提升性能
        feat = frozen_encoder(cycles_tensor.to(device))
    return feat.cpu()

def load_data(weight_path, device, train_dir, local_num_cycles, global_num_cycles, eps=1e-8):
    """加载 LoRA 标签权重并读取对应的原始电池 Cycle 数据"""
    weights = torch.load(weight_path, map_location=device)
    param_A = weights["param_A"].to(device).float()
    param_B = weights["param_B"].to(device).float()

    # 计算归一化 A, B 和 log_scale
    norm_A = torch.norm(param_A, p="fro")
    norm_B = torch.norm(param_B, p="fro")
    A_norm = param_A / (norm_A + eps)
    B_norm = param_B / (norm_B + eps)
    log_scale = torch.log(norm_A * norm_B + eps).unsqueeze(0)

    # 解析路径获取电池数据
    parts = weight_path.stem.split("_")
    start_idx = int(parts[-1])
    battery_name = "_".join(parts[:-1])
    battery_path = train_dir / f"{battery_name}.pkl"
    battery = BatteryData.load(str(battery_path))

    # 提取 Cycle 信号
    g_data = battery.cycle_data[0 : global_num_cycles]
    global_cycles = np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in g_data])
    
    l_data = battery.cycle_data[start_idx : start_idx + local_num_cycles]
    local_cycles = np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in l_data])

    return A_norm.cpu(), B_norm.cpu(), log_scale.cpu(), \
           torch.tensor(local_cycles), torch.tensor(global_cycles)

# ==========================================
# 2. 训练相关函数 (由 Main 调用)
# ==========================================

def hyper_lora_loss(pred_A, pred_B, pred_log_s, target_A, target_B, target_log_s):
    """独立的解耦 Loss 计算函数"""
    criterion = nn.MSELoss()
    loss_A = criterion(pred_A, target_A)
    loss_B = criterion(pred_B, target_B)
    loss_S = criterion(pred_log_s, target_log_s)
    total_loss = loss_A + loss_B + 10.0 * loss_S
    return total_loss, (loss_A.item(), loss_B.item(), loss_S.item())

class HyperCacheDataset(Dataset):
    """带缓存的 Dataset，防止训练时重复跑 Encoder"""
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def prepare_hyper_dataset(files, encoder, train_dir, local_num, global_num, device):
    """预计算所有特征并打包为 Dataset"""
    cached_data = []
    for file in tqdm(files, desc="Pre-computing Features"):
        A_n, B_n, l_s, l_cyc, g_cyc = load_data(file, device, train_dir, local_num, global_num)
        
        # 提取特征
        feat_l = build_X_from_cycles(l_cyc, encoder, device)
        feat_g = build_X_from_cycles(g_cyc, encoder, device)
        
        cached_data.append((feat_g, feat_l, A_n, B_n, l_s))
    return HyperCacheDataset(cached_data)

def train(train_loader, model_save_path, device, epochs=10000, early_stop_patience=50):
    """
    超网络训练逻辑：
    1) 在函数内完成 K 折交叉验证
    2) 在 loss 中加入 L2 正则化
    3) 学习率衰减与早停均基于 CV 均值验证损失
    """
    full_dataset = train_loader.dataset
    num_samples = len(full_dataset)
    if num_samples < 2:
        raise ValueError("训练样本数不足，至少需要 2 个样本以执行交叉验证。")

    # 内置超参数（按用户要求全部放在 train 函数中，不新增 argparse 参数）
    num_folds = min(5, num_samples)
    batch_size = train_loader.batch_size or 64
    lambda_l2 = 1e-6

    # 构建 K 折索引
    all_indices = np.arange(num_samples)
    rng = np.random.default_rng(42)
    rng.shuffle(all_indices)
    fold_indices = np.array_split(all_indices, num_folds)

    model = HyperLoRAGenerator(feature_dim=64, rank=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    # 学习率衰减：当验证集 loss 20 个 epoch 不下降时，降低 LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    best_val_loss = float('inf')
    no_improve = 0
    
    for epoch in range(epochs):
        fold_train_losses = []
        fold_val_losses = []

        # 每个 epoch 遍历 K 折：每折用 K-1 份训练、1 份验证
        for k in range(num_folds):
            val_idx = fold_indices[k]
            train_idx = np.concatenate([fold_indices[i] for i in range(num_folds) if i != k])

            train_subset = Subset(full_dataset, train_idx.tolist())
            val_subset = Subset(full_dataset, val_idx.tolist())

            fold_train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            fold_val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # --- 该折训练 ---
            model.train()
            train_loss = 0.0
            train_batches = 0
            for feat_g, feat_l, t_A, t_B, t_s in fold_train_loader:
                feat_g, feat_l = feat_g.to(device), feat_l.to(device)
                t_A, t_B, t_s = t_A.to(device), t_B.to(device), t_s.to(device)

                optimizer.zero_grad()
                p_A, p_B, p_s = model(feat_g, feat_l)

                base_loss, _ = hyper_lora_loss(p_A, p_B, p_s, t_A, t_B, t_s)
                l2_penalty = torch.zeros(1, device=device)
                for p in model.parameters():
                    l2_penalty = l2_penalty + torch.sum(p.pow(2))
                loss = base_loss + lambda_l2 * l2_penalty

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            fold_train_losses.append(train_loss / max(train_batches, 1))

            # --- 该折验证 ---
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for feat_g, feat_l, t_A, t_B, t_s in fold_val_loader:
                    feat_g, feat_l = feat_g.to(device), feat_l.to(device)
                    t_A, t_B, t_s = t_A.to(device), t_B.to(device), t_s.to(device)

                    p_A, p_B, p_s = model(feat_g, feat_l)
                    v_loss, _ = hyper_lora_loss(p_A, p_B, p_s, t_A, t_B, t_s)
                    val_loss += v_loss.item()
                    val_batches += 1

            fold_val_losses.append(val_loss / max(val_batches, 1))

        avg_train_loss = float(np.mean(fold_train_losses))
        avg_val_loss = float(np.mean(fold_val_losses))

        # 打印进度
        if (epoch + 1) % 5 == 0:
            logging.info(
                f"Epoch {epoch+1} | CV-Train Loss: {avg_train_loss:.6f} | CV-Val Loss: {avg_val_loss:.6f}"
            )

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 早停判断
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved at epoch {epoch+1}")
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            logging.info(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
            break

    logging.info("Training process completed.")


# ==========================================
# 4. 测试相关函数 (权重生成准确率测试)
# ==========================================

def test(hyper_model_path, model_path, weights_dir, labeled_dir, device, local_num_cycles, global_num_cycles, result_path):
    """
    测试逻辑：
    直接对比 HyperLoRA 生成的 (A, B, log_s) 与 pkl 文件中真值的 MSE。
    """
    # 1. 加载模型
    encoder = load_frozen_encoder(model_path, device)
    hyper_gen = HyperLoRAGenerator(feature_dim=64, rank=4).to(device)
    hyper_gen.load_state_dict(torch.load(hyper_model_path, map_location=device))
    hyper_gen.eval()

    criterion = nn.MSELoss()
    results = []

    # 2. 遍历所有 test_ 开头的文件夹
    test_subdirs = sorted([d for d in weights_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])
    
    for subdir in test_subdirs:
        weight_files = sorted(subdir.glob('*.pkl'))
        if not weight_files: continue
        
        # 对应原始数据的训练目录名称
        current_labeled_dir = labeled_dir / subdir.name

        total_loss_A, total_loss_B, total_loss_S = 0, 0, 0
        count = 0

        for f in tqdm(weight_files, desc=f"Testing {subdir.name}"):
            # 加载真值和原始信号
            t_A, t_B, t_s, l_cyc, g_cyc = load_data(f, device, current_labeled_dir, 
                                                    local_num_cycles, global_num_cycles)
            
            # 提取特征
            feat_l = build_X_from_cycles(l_cyc, encoder, device).unsqueeze(0).to(device)
            feat_g = build_X_from_cycles(g_cyc, encoder, device).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 生成预测值
                p_A, p_B, p_s = hyper_gen(feat_g, feat_l)
                
                # 计算针对组件的 MSE (需要搬运真值到 device)
                loss_A = criterion(p_A, t_A.unsqueeze(0).to(device)).item()
                loss_B = criterion(p_B, t_B.unsqueeze(0).to(device)).item()
                loss_S = criterion(p_s, t_s.unsqueeze(0).to(device)).item()

                total_loss_A += loss_A
                total_loss_B += loss_B
                total_loss_S += loss_S
                count += 1

        if count > 0:
            avg_A = total_loss_A / count
            avg_B = total_loss_B / count
            avg_S = total_loss_S / count
            avg_total = avg_A + avg_B + avg_S
            
            results.append([subdir.name, avg_A, avg_B, avg_S, avg_total])
            logging.info(f"Result for {subdir.name} | MSE_A: {avg_A:.6f}, MSE_S: {avg_S:.6f}")

    # 3. 保存结果
    header = ['dataset', 'mse_A', 'mse_B', 'mse_log_s', 'total_mse']
    df = pd.DataFrame(results, columns=header)
    df.to_csv(result_path, index=False)
    logging.info(f"Accuracy test finished. Saved to {result_path}")

# ==========================================
# 3. Main 宏观调用
# ==========================================

if __name__ == "__main__":
    # 1. 配置与参数
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="main")
    config = OmegaConf.create(config)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide",
        choices=['True', 'False'],
        help="Whether to use sliding window training (string True/False)",
        default='True'
    )
    parser.add_argument("--window_size", type=int, default=200)    
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--local_num_cycles", type=int, default=5)
    parser.add_argument("--global_num_cycles", type=int, default=20)
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_test", action='store_true', default=True)
    args = parser.parse_args()
    # args.do_train = False

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 2. 路径对齐
    labeled_dir = Path(config.path.labeled_dir)
    result_dir = Path(config.path.results_dir) / f"HyperLoRA_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(config.path.results_dir) / 'GMA-NET.pkl'
    if not args.slide:
        weights_dir = Path(config.path.weights_dir) / 'no_slide'
        result_path = result_dir / f'HyperLoRA_no_slide_{args.local_num_cycles}_{args.global_num_cycles}_test.csv'
        hyper_model_path = Path(config.path.results_dir) / f'HyperLoRA_{args.local_num_cycles}_{args.global_num_cycles}_no_slide.pkl'
    else:
        weights_dir = Path(config.path.weights_dir) / f'wsize_{args.window_size}_stride_{args.stride}'
        result_path = result_dir / f'HyperLoRA_wsize_{args.local_num_cycles}_{args.global_num_cycles}_{args.window_size}_stride_{args.stride}_test.csv'
        hyper_model_path = Path(config.path.results_dir) / f'HyperLoRA_{args.local_num_cycles}_{args.global_num_cycles}_wsize_{args.window_size}_stride_{args.stride}.pkl'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A. 训练宏观逻辑
    if args.do_train:
        logging.info("--- Starting HyperLoRA Training ---")
        encoder = load_frozen_encoder(model_path, device)
        # 只用 train 文件夹下的权重训练
        train_weight_files = sorted((weights_dir / 'train').glob('*.pkl'))
        dataset = prepare_hyper_dataset(train_weight_files, encoder, labeled_dir / 'train', 
                                       args.local_num_cycles, args.global_num_cycles, device)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        train(train_loader, hyper_model_path, device)

    # B. 测试宏观逻辑 (计算 LoRA 生成准确率)
    if args.do_test:
        logging.info("--- Starting HyperLoRA Accuracy Test ---")
        test(
            hyper_model_path, 
            model_path, 
            weights_dir, 
            labeled_dir, 
            device, 
            args.local_num_cycles, 
            args.global_num_cycles,
            result_path,
        )