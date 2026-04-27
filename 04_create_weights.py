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


def combined_loss(preds, targets, param_A, param_B, 
                  lambda_smooth=0.01,    # 时间平滑：相邻窗口不要突变
                  lambda_soh=0.01,       # 状态一致：相似SOH的权重靠近 (建议调小点，给流动留空间)
                  lambda_reg=1e-2,       # L2正则：控制数值范围在合理区间 (防止-100~100)
                  lambda_orth=0.1,       # 正交惩罚：提高信息密度，固定参数含义
                  lambda_dir=0.05,       # 退化斜率：强迫所有电池向同一个方向“流动”
                  sigma_soh=0.005):      # SOH核带宽：收窄以提高局部灵敏度
    """
    preds: [num_wins, window_size, 1]
    targets: [num_wins, window_size] (SOH标签)
    param_A: [num_wins, 64, rank]
    param_B: [num_wins, rank, 32]
    """
    # 1. 预测损失 (L_pred)
    l_pred = nn.MSELoss()(preds.squeeze(-1), targets)

    # 2. 计算当前权重矩阵 W: [num_wins, 64, 32]
    W_curr = torch.bmm(param_A, param_B)
    num_wins = W_curr.size(0)
    device = W_curr.device

    # 3. 时间平滑损失 (L_smooth)
    l_smooth = 0
    if num_wins > 1:
        diff_W = W_curr[1:] - W_curr[:-1]
        l_smooth = torch.mean(diff_W**2)

    # 4. SOH 状态一致性损失 (L_soh)
    l_soh = 0
    soh_centers = targets.mean(dim=1)  # [num_wins]
    if num_wins > 1:
        dist_soh = torch.abs(soh_centers.unsqueeze(0) - soh_centers.unsqueeze(1))
        K_soh = torch.exp(-(dist_soh**2) / (2 * sigma_soh**2))
        W_flat = W_curr.view(num_wins, -1)
        dist_W = torch.cdist(W_flat, W_flat, p=2)**2
        l_soh = torch.mean(K_soh * dist_W)

    # 5. 【新增】退化斜率约束 (L_direction)
    # 目的：让 dW/dSOH 的向量在所有窗口间尽量一致
    l_direction = 0
    if num_wins > 1:
        delta_W = W_curr[1:] - W_curr[:-1]  # [num_wins-1, 64, 32]
        # 计算 SOH 的变化量，注意加 epsilon 防止除以 0
        delta_SOH = (soh_centers[1:] - soh_centers[:-1]).view(-1, 1, 1)
        # 计算斜率向量：单位 SOH 变化引起的 W 变化
        W_slope = delta_W / (delta_SOH + 1e-6) 
        # 约束：所有步长的斜率向量与其平均斜率的差异（即减小斜率的方差）
        # 这样会强迫所有电池在参数空间内沿平行线移动
        avg_slope = torch.mean(W_slope, dim=0, keepdim=True)
        l_direction = torch.mean((W_slope - avg_slope)**2)

    # 6. 【新增】正交化惩罚 (L_orth)
    # 目的：防止参数冗余，确保 rank 个维度各司其职
    rank = param_A.size(-1)
    I = torch.eye(rank, device=device).unsqueeze(0)
    # A 的列正交: A^T @ A ≈ I
    l_orth_A = torch.mean((torch.bmm(param_A.transpose(1, 2), param_A) - I)**2)
    # B 的行正交: B @ B^T ≈ I
    l_orth_B = torch.mean((torch.bmm(param_B, param_B.transpose(1, 2)) - I)**2)
    l_orth = l_orth_A + l_orth_B

    # 7. 强化正则项 (L_reg)
    l_reg = torch.mean(param_A**2) + torch.mean(param_B**2)

    # 总损失组合
    total_loss = l_pred + \
                 lambda_smooth * l_smooth + \
                 lambda_soh * l_soh + \
                 lambda_reg * l_reg + \
                 lambda_orth * l_orth + \
                 lambda_dir * l_direction

    return total_loss


def canonicalize_lora(param_A, param_B, eps=1e-8):
    """
    将 A/B 规范化并提取 log_s，减少尺度不确定性。
    同时做简单符号对齐，降低等价分解带来的标签抖动。
    """
    norm_A = torch.norm(param_A, p="fro")
    norm_B = torch.norm(param_B, p="fro")

    A_norm = param_A / (norm_A + eps)
    B_norm = param_B / (norm_B + eps)
    log_s = torch.log(norm_A * norm_B + eps).view(1)

    sign = torch.sign(A_norm.sum(dim=0))
    sign[sign == 0] = 1.0
    A_norm = A_norm * sign.view(1, -1)
    B_norm = B_norm * sign.view(-1, 1)

    return A_norm, B_norm, log_s

def train(windows, model, device, sub_weights_dir, battery_id, num_restarts=300):
    """
    极致精度版：300次并行初始化选优
    """
    num_wins = len(windows)
    if num_wins == 0: return

    # 1. 预提取特征 (保持不变，避免重复计算)
    all_X = torch.tensor(np.stack([w['X'] for w in windows]), dtype=torch.float32).to(device)
    all_y = torch.tensor(np.stack([w['y'] for w in windows]), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        flat_X = all_X.view(-1, all_X.shape[-1])
        all_features = model.encoder(flat_X).view(num_wins, all_X.shape[1], -1)

    # 2. 海选阶段 (Phase A: The Race)
    best_init_loss = float('inf')
    best_seeds = None
    
    # 为了显存安全，建议分批进行海选 (比如每批 50 个)
    batch_size_restarts = 50 
    for b in range(0, num_restarts, batch_size_restarts):
        current_batch_size = min(batch_size_restarts, num_restarts - b)
        
        # 批量初始化参数 [batch_restarts, num_wins, 64, rank]
        # 注意：这里我们可以利用广播机制一次性计算多个种子的 Loss
        # 但为了代码可读性和稳定性，我们用循环寻找最有潜力的种子
        for i in range(current_batch_size):
            # 随机正交初始化
            temp_A = torch.empty(num_wins, 64, model.rank, device=device)
            temp_B = torch.empty(num_wins, model.rank, 32, device=device)
            nn.init.orthogonal_(temp_A)
            nn.init.orthogonal_(temp_B)
            
            temp_A = nn.Parameter(temp_A)
            temp_B = nn.Parameter(temp_B)
            temp_opt = optim.Adam([temp_A, temp_B], lr=1e-2)
            
            # 短冲刺：30个 epoch 足以看出收敛潜力
            for _ in range(30):
                temp_opt.zero_grad()
                W = torch.bmm(temp_A, temp_B)
                p = model.head(torch.matmul(all_features, W))
                l = combined_loss(p, all_y, temp_A, temp_B)
                l.backward()
                temp_opt.step()
            
            this_loss = l.item()
            if this_loss < best_init_loss:
                best_init_loss = this_loss
                best_seeds = (temp_A.detach().clone(), temp_B.detach().clone())

    # 3. 决赛阶段 (Phase B: Fine-tuning)
    param_A = nn.Parameter(best_seeds[0])
    param_B = nn.Parameter(best_seeds[1])
    optimizer = optim.Adam([param_A, param_B], lr=5e-3)
    
    # 3. 联合优化循环
    max_epochs = 1200
    patience = 50
    best_loss = float('inf')
    counter = 0

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        # 计算所有窗口对应的预测值
        W_curr = torch.bmm(param_A, param_B) # [num_wins, 64, 32]
        # 应用动态权重: 特征 [N, L, 64] @ 权重 [N, 64, 32] -> [N, L, 32]
        win_out = torch.matmul(all_features, W_curr)
        preds = model.head(win_out) # [num_wins, window_size, 1]

        # 调用上面定义的组合 Loss
        loss = combined_loss(preds, all_y, param_A, param_B)
        
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-7:
            best_loss = loss.item()
            counter = 0
            best_A = param_A.detach().clone()
            best_B = param_B.detach().clone()
        else:
            counter += 1
        
        if counter >= patience:
            break

    # 4. 保存每个窗口的结果：仅保存 A/B/log_s
    for i in range(num_wins):
        start_idx = windows[i]['start_idx']
        save_path = sub_weights_dir / f'{battery_id}_{start_idx}.pkl'

        A_norm, B_norm, log_s = canonicalize_lora(best_A[i], best_B[i])

        torch.save({
            'start_idx': start_idx,
            'param_A': A_norm.cpu(),
            'param_B': B_norm.cpu(),
            'param_log_s': log_s.cpu()
        }, save_path)
    
    return best_A, best_B


# -----------------------------------------------------------------------------
# 2. 补全 Test 函数: 双重测试 (窗口 + 全电池)
# -----------------------------------------------------------------------------
def test(weight_path, model, device, battery, slide, start_idx, window_size, test_result_path):
    """
    推理时优先使用 A/B/log_s 还原权重，兼容历史 weight_W。
    """
    
    # 1. 加载合成后的权重 W [64, 32]
    # ---------------------------------------------------------
    weights = torch.load(weight_path, map_location=device)
    
    p_A = weights['param_A'].to(device).float()
    p_B = weights['param_B'].to(device).float()
    p_log_s = weights.get('param_log_s', torch.zeros(1)).to(device).float().view(1)
    scale = torch.exp(torch.clamp(p_log_s, min=-5.0, max=5.0)).view(1, 1)
    W = scale * torch.matmul(p_A, p_B)
    
    # 2. 准备数据
    # ---------------------------------------------------------
    cycles = battery.cycle_data
    if not slide:
        win_cycles = cycles
    else:
        end_idx = start_idx + window_size
        win_cycles = cycles[start_idx:end_idx]
    
    X_win = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in win_cycles]), dtype=torch.float32).to(device)
    y_win = np.array([c.labeled_soh for c in win_cycles], dtype=np.float32)
    
    X_bat = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in cycles]), dtype=torch.float32).to(device)
    y_bat = np.array([c.labeled_soh for c in cycles], dtype=np.float32)
    
    # 3. 推理 (Inference)
    # ---------------------------------------------------------
    model.eval()
    with torch.no_grad():
        # A. 提取特征
        feat_win = model.encoder(X_win) # [L_win, 64]
        feat_bat = model.encoder(X_bat) # [L_bat, 64]
        
        # B. 直接使用 W 进行矩阵乘法 (feat @ W)
        # 注意：这里 W 已经是 [64, 32]，所以直接乘即可
        aligned_win = torch.matmul(feat_win, W) # [L_win, 32]
        aligned_bat = torch.matmul(feat_bat, W) # [L_bat, 32]
        
        # C. 预测 SOH
        pred_win = model.head(aligned_win).cpu().squeeze(1).numpy()
        pred_bat = model.head(aligned_bat).cpu().squeeze(1).numpy()
        
    # 4. 计算指标与写入 CSV (保持原样)
    # ---------------------------------------------------------
    def calc_metrics(pred, true):
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(mse)
        return mse, mae, rmse

    mse_win, mae_win, rmse_win = calc_metrics(pred_win, y_win)
    mse_bat, mae_bat, rmse_bat = calc_metrics(pred_bat, y_bat)
    
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
    results_dir = Path(config.path.create_weights_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pretrained_path = Path(config.path.models_dir) / 'GMA-NET.pkl'
    if not slide:
        weights_dir = Path(config.path.weights_dir) / 'full'
        result_file_name = 'GMANet_full_test'
    else:
        weights_dir = Path(config.path.weights_dir) / f'wsize_{window_size}_stride_{stride}'
        result_file_name = f'GMANet_wsize{window_size}_stride{stride}'
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_model(pretrained_path, device, rank=4)

    data_dirs = [d for d in sorted(labeled_dir.iterdir()) if d.is_dir()]
    # 循环电池文件夹
    for data_dir in data_dirs:
        files = sorted(data_dir.glob("*.pkl"))
        sub_weights_dir = weights_dir / data_dir.name
        sub_weights_dir.mkdir(parents=True, exist_ok=True)

        for file in tqdm(files, desc=f'Processing {data_dir.name}'):
            battery = BatteryData.load(str(file))
            # 1. 划分该电池的所有窗口
            windows = slide_cycle(battery, slide, window_size, stride)
            if not windows: continue

            # 2. 联合训练这颗电池的所有窗口 (得到平滑权重)
            train(windows, model, device, sub_weights_dir, file.stem)

            # 3. 测试与记录结果 (循环所有窗口进行测试记录)
            result_path = results_dir / f'{result_file_name}_{data_dir.name}.csv'
            for i in range(len(windows)):
                start_idx = windows[i]['start_idx']
                weights_path = sub_weights_dir / f'{file.stem}_{start_idx}.pkl'
                test(weights_path, model, device, battery, slide, start_idx, window_size, result_path)