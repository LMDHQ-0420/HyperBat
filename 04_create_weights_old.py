import logging
import os
import shutil
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import argparse

from battery_data import BatteryData
from model import GMANet


def load_frozen_model(pretrained_path, device, rank=4):
    """
    加载预训练模型，并迁移权重到 GMANet，冻结骨架。
    """
    # 1. 实例化 Phase 2 模型
    model = GMANet(rank=rank).to(device)
        
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()
    
    # 3. 过滤掉不匹配的键 (主要是 temp_connector)
    # 仅保留 Encoder 和 Head 的参数
    filtered_dict = {k: v for k, v in pretrained_dict.items() 
                     if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    # 4. 冻结 Encoder 和 Head
    model.eval() # 设为 eval 模式 (关闭 Dropout/BatchNorm 更新)
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = False
    
    logging.info("Model loaded. Encoder and Head are FROZEN.")

    return model

def slide_cycle(battery, slide, window_size, stride):
    """
    对单个电池进行滑动窗口切分
    """
    if not slide:
        return [{
            'X': np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in battery.cycle_data]),
            'y': np.array([c.labeled_soh for c in battery.cycle_data]),
            'start_idx': 0,
        }]

    cycles = battery.cycle_data
    num_cycles = len(cycles)
    windows = []

    # 如果电池总圈数小于窗口，则跳过或取全部 (这里选择跳过)
    if num_cycles < window_size:
        return []

    for start_idx in range(0, num_cycles - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_cycles = cycles[start_idx:end_idx]
        
        # 提取 QC (X) 和 SOH (y)
        qc_list = [np.asarray(c.labeled_Qc, dtype=np.float32) for c in window_cycles]
        soh_list = [c.labeled_soh for c in window_cycles]
        
        windows.append({
            'X': np.stack(qc_list),      # [200, 400]
            'y': np.array(soh_list),     # [200]
            'start_idx': start_idx,
        })
        
    return windows

def train(X, y, model, device, model_path):
    """
    针对特定窗口数据 X, y，优化模型中的 A, B 参数，并保存结果。
    注意：model 的 Encoder 和 Head 必须是冻结状态。
    """
    
    # 1. 初始化 A 和 B (需要梯度)
    # ---------------------------------------------------------
    # 使用正交初始化加速收敛
    # A: [64, rank], B: [rank, 32]
    # 我们将它们包装成 nn.Parameter 以便放入优化器，但它们不是 model 的一部分
    # 或者是 model 的临时参数。这里我们作为独立张量优化。
    param_A = torch.empty(model.enc_dim, model.rank, device=device, requires_grad=True)
    param_B = torch.empty(model.rank, model.head_in_dim, device=device, requires_grad=True)
    nn.init.orthogonal_(param_A)
    nn.init.orthogonal_(param_B)
    
    # 2. 准备数据
    # ---------------------------------------------------------
    # 因为每个窗口数据量很少 (200条)，直接全量放入 GPU 训练即可，不需要 DataLoader
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    
    # 3. 定义优化器
    # ---------------------------------------------------------
    # 只优化 param_A 和 param_B
    lr=1e-2
    optimizer = optim.Adam([param_A, param_B], lr) # 学习率相对大一点，因为只有两层线性参数
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=20,
        threshold=1e-4,
        min_lr=1e-6,
    )
    
    # 4. 训练循环 (Overfitting this window)
    # ---------------------------------------------------------
    model.eval() # 确保 Dropout/BN 关闭 (虽然 Encoder 冻结了，但以防万一)
    max_epochs = 1000
    best_loss = float('inf')
    no_improve = 0
    early_stop_patience = 50  # 与 03 保持一致的早停策略

    # 保存最佳参数副本用于最终落盘
    best_A = param_A.detach().clone()
    best_B = param_B.detach().clone()

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        # Forward: 传入动态参数 A, B
        preds = model(X_tensor, param_A, param_B)
        
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        scheduler.step(loss_val)

        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            no_improve = 0
            best_A = param_A.detach().clone()
            best_B = param_B.detach().clone()
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            break
        
    # 5. 保存结果
    # ---------------------------------------------------------
    # 我们只保存 A 和 B，因为 Encoder/Head 是通用的不需要重复保存
    # 这样可以节省大量空间
    save_dict = {
        'param_A': best_A.cpu(),
        'param_B': best_B.cpu()
    }
    torch.save(save_dict, model_path)


# -----------------------------------------------------------------------------
# 2. 补全 Test 函数: 双重测试 (窗口 + 全电池)
# -----------------------------------------------------------------------------
def test(weight_path, model, device, battery, slide, start_idx, window_size, test_result_path):
    """
    使用保存的 A, B 参数，测试:
    1. 当前窗口 (mse_win)
    2. 整个电池生命周期 (mse_bat)
    并将结果写入 CSV。
    """
    
    # 1. 加载刚刚训练好的 A, B
    # ---------------------------------------------------------
    weights = torch.load(weight_path, map_location=device)
    param_A = weights['param_A'].to(device)
    param_B = weights['param_B'].to(device)
    
    # 2. 准备数据
    # ---------------------------------------------------------
    # A. 当前窗口数据 (Window Data)
    # 我们需要根据 start_idx 重新切一下，或者传入 (为了简单，这里重切)
    cycles = battery.cycle_data
    if not slide:
        win_cycles = cycles
    else:
        end_idx = start_idx + window_size
        win_cycles = cycles[start_idx:end_idx]
    
    X_win = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in win_cycles]), dtype=torch.float32).to(device)
    y_win = np.array([c.labeled_soh for c in win_cycles], dtype=np.float32)
    
    # B. 全电池数据 (Battery Data)
    X_bat = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in cycles]), dtype=torch.float32).to(device)
    y_bat = np.array([c.labeled_soh for c in cycles], dtype=np.float32)
    
    # 3. 推理 (Inference)
    # ---------------------------------------------------------
    model.eval()
    with torch.no_grad():
        # 预测窗口
        pred_win = model(X_win, param_A, param_B).cpu().squeeze(1).numpy()
        # 预测全电池 (用针对该窗口优化的参数去预测全生命周期，看泛化性)
        pred_bat = model(X_bat, param_A, param_B).cpu().squeeze(1).numpy()
        
    # 4. 计算指标
    # ---------------------------------------------------------
    def calc_metrics(pred, true):
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(mse)
        return mse, mae, rmse

    mse_win, mae_win, rmse_win = calc_metrics(pred_win, y_win)
    mse_bat, mae_bat, rmse_bat = calc_metrics(pred_bat, y_bat)
    
    # 5. 写入 CSV
    # ---------------------------------------------------------
    # header = ['battery', 'window_start', 'mse_win', 'mae_win', 'rmse_win', 'mse_bat', 'mae_bat', 'rmse_bat']
    
    test_result_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not test_result_path.exists()
    
    with open(test_result_path, 'a') as f:
        if need_header:
            header = ['battery', 'window_start', 'mse_win', 'mae_win', 'rmse_win', 'mse_bat', 'mae_bat', 'rmse_bat']
            f.write(','.join(header) + '\n')
            
        line = f"{battery.cell_id},{start_idx},{mse_win:.6f},{mae_win:.6f},{rmse_win:.6f},{mse_bat:.6f},{mae_bat:.6f},{rmse_bat:.6f}\n"
        f.write(line)




if __name__ == "__main__":
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
    parser.add_argument(
        "--window_size",
        type=int,
        help="Window size for sliding window training",
        default=200
    )    
    parser.add_argument(
        "--stride",
        type=int,
        help="Stride for sliding window training",
        default=50
    )
    args = parser.parse_args()

    slide = args.slide == 'True'
    window_size = args.window_size
    stride = args.stride

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Slide mode: {slide} | window_size={window_size}, stride={stride}")

    labeled_dir = Path(config.path.labeled_dir)
    result_dir = Path(config.path.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    pretrained_path = result_dir / 'GMA-NET.pkl'
    if not slide:
        weights_dir = Path(config.path.weights_dir) / 'no_slide'
        result_file_name = 'GMANet_no_slide_test'
    else:
        weights_dir = Path(config.path.weights_dir) / f'wsize_{window_size}_stride_{stride}'
        result_file_name = f'GMANet_wsize_{window_size}_stride_{stride}_test'
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_model(pretrained_path, device, rank=4)

    data_dirs = [d for d in sorted(labeled_dir.iterdir()) if d.is_dir()]
    for data_dir in data_dirs:
        files = sorted(data_dir.glob("*.pkl"))

        sub_weights_dir = weights_dir / data_dir.name
        sub_weights_dir.mkdir(parents=True, exist_ok=True)

        for file in tqdm(files, desc=f'Loading {data_dir}', total=len(files)):
            battery = BatteryData.load(str(file))
            windows = slide_cycle(battery, slide, window_size, stride)
            for win in tqdm(windows, desc=f'Processing {file.name}', leave=False):
                start_idx = win['start_idx']
                X = win['X']
                y = win['y']

                weights_path = sub_weights_dir / f'{file.stem}_{start_idx}.pkl'
                train(X, y, model, device, weights_path)

                result_path = result_dir / f'{result_file_name}_{data_dir.name}.csv'
                test(weights_path, model, device, battery, slide, start_idx, window_size, result_path)