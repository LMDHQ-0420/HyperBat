import logging
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from battery_data import BatteryData
from model import GMANetPreTrain


import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter


def read_dataset(train_dir):
    """
    读取文件列表，并根据电池体系计算采样权重
    Args:
        file_list: 包含 .pkl 路径的列表 (由上一轮的划分函数生成)
    Returns:
        X: 输入张量
        y: 标签张量
        samples_weights: 每个样本对应的采样权重 (Tensor)
    """
    all_qc = []
    all_soh = []
    all_chem_labels = [] # 用于记录每条数据是哪个体系 (e.g., 'LFP', 'NCM')

    # 1. 读取数据并记录体系
    # ---------------------------------------------------------
    files = sorted(train_dir.glob("*.pkl"))

    for file in tqdm(sorted(train_dir.glob("*.pkl")), desc=f'Loading {train_dir}', total=len(files)):
        battery = BatteryData.load(str(file))
        chem = battery.cathode_material 
        
        # 提取 Cycle 数据
        cycles_qc = [np.asarray(c.labeled_Qc, dtype=np.float32) for c in battery.cycle_data]
        cycles_soh = [c.labeled_soh for c in battery.cycle_data]

        all_qc.extend(cycles_qc)
        all_soh.extend(cycles_soh)
        all_chem_labels.extend([chem] * len(cycles_qc))

    # 2. 计算权重 (核心逻辑)
    # ---------------------------------------------------------
    # 统计每个体系的总样本数 (Cycle 级别)
    # 例如: {'LFP': 10000, 'NMC': 200, 'LCO': 100}
    chem_counts = Counter(all_chem_labels)
    print(f"Dataset Distribution (Cycles): {chem_counts}")

    # 计算每个体系的权重 = 1.0 / 数量
    # 数量越少，权重越大
    chem_weights = {chem: 1.0 / count for chem, count in chem_counts.items()}
    
    # 为每一个样本 (Cycle) 分配权重
    # samples_weights 的长度等于 len(all_qc)
    samples_weights = [chem_weights[chem] for chem in all_chem_labels]
    samples_weights = torch.tensor(samples_weights, dtype=torch.double)

    # 3. 转换数据张量
    # ---------------------------------------------------------
    X = torch.tensor(np.stack(all_qc), dtype=torch.float32)
    y = torch.tensor(all_soh, dtype=torch.float32).unsqueeze(1)
    return X, y, samples_weights




# def train(X, y, samples_weights, model_path, l2_weight_decay=1e-5):

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. 创建 Sampler (关键修改)
#     # replacement=True 表示允许重复采样 (这是平衡数据的关键)
#     sampler = WeightedRandomSampler(
#         weights=samples_weights,
#         num_samples=len(samples_weights),
#         replacement=True
#     )

#     # 2. 创建 DataLoader
#     train_ds = TensorDataset(X, y)
#     train_loader = DataLoader(
#         train_ds, 
#         batch_size=128,      # 建议调大一点，比如 64 或 128，让每个 Batch 更容易混入少样本
#         sampler=sampler,    # 注入采样器
#         shuffle=False       # 必须为 False ！！！
#     )

#     model = GMANetPreTrain().to(device)
    
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=1e-3,
#         weight_decay=l2_weight_decay,
#     )
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',
#         factor=0.1,
#         patience=20,
#         threshold=1e-4,
#         min_lr=1e-6,
#     )

#     # 训练，最多 10000 个 epoch，包含学习率衰减与早停
#     max_epochs = 10000
#     best_loss = float('inf')
#     no_improve = 0
#     early_stop_patience = 50  # 连续 50 个 epoch 无提升则早停

#     for _ in range(max_epochs):
#         model.train()
#         running_loss = 0.0
#         batch_count = 0
#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
           
#             optimizer.zero_grad()
#             pred = model(xb)
#             loss = criterion(pred, yb)
            
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             batch_count += 1

#         epoch_loss = running_loss / max(batch_count, 1)

#         logging.info(f'Epoch {_+1}/{max_epochs}, Loss: {epoch_loss:.6f}')

#         if epoch_loss < best_loss - 1e-6:
#             best_loss = epoch_loss
#             no_improve = 0
#             logging.info('Best model so far.')
#         else:
#             no_improve += 1

#         # 调度学习率（基于 loss 的 plateau）
#         scheduler.step(epoch_loss)

#         # 早停判断
#         if no_improve >= early_stop_patience:
#             logging.info('Early stopping: no improvement for patience limit.')
#             break

#     # 训练完成后保存一次模型权重
#     torch.save(model.state_dict(), model_path)
#     logging.info(f'Training completed. Model weights saved to {model_path}.')


def train(X, y, samples_weights, model_path, abs_path, l2_weight_decay=1e-5, lambda_orth_p1=0.01):
    """
    Args:
        lambda_orth_p1: 正交化损失的系数，建议 0.01 ~ 0.1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建 Sampler
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )

    # 2. 创建 DataLoader
    train_ds = TensorDataset(X, y)
    train_loader = DataLoader(
        train_ds, 
        batch_size=128,
        sampler=sampler,
        shuffle=False
    )

    model = GMANetPreTrain().to(device)
    
    # --- 建议：手动进行正交初始化，给模型一个好的起点 ---
    model.temp_connector.reset_parameters()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=l2_weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=20,
        threshold=1e-4,
        min_lr=1e-6,
    )

    # 训练，最多 10000 个 epoch，包含学习率衰减与早停
    max_epochs = 10000
    best_loss = float('inf')
    no_improve = 0
    early_stop_patience = 50  # 连续 50 个 epoch 无提升则早停

    for epoch in range(max_epochs):
        model.train()
        running_mse = 0.0
        running_orth = 0.0
        batch_count = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
           
            optimizer.zero_grad()
            
            # 1. 前向预测
            pred = model(xb)
            mse_loss = criterion(pred, yb)
            
            # 2. LoRA 正交约束
            param_A = model.temp_connector.param_A.unsqueeze(0)
            param_B = model.temp_connector.param_B.unsqueeze(0)
            rank = param_A.size(-1)
            I = torch.eye(rank, device=device).unsqueeze(0)
            l_orth_A = torch.mean((torch.bmm(param_A.transpose(1, 2), param_A) - I)**2)
            l_orth_B = torch.mean((torch.bmm(param_B, param_B.transpose(1, 2)) - I)**2)
            orth_loss = l_orth_A + l_orth_B
            # 3. 组合总损失
            total_loss = mse_loss + lambda_orth_p1 * orth_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_mse += mse_loss.item()
            running_orth += orth_loss.item()
            batch_count += 1

        epoch_mse = running_mse / max(batch_count, 1)
        epoch_orth = running_orth / max(batch_count, 1)
        epoch_total = epoch_mse + lambda_orth_p1 * epoch_orth

        logging.info(f'Epoch {epoch+1}, Total: {epoch_total:.6f}, MSE: {epoch_mse:.6f}, Orth: {epoch_orth:.6f}')

        # 注意：早停和调度器建议基于 epoch_total 或 epoch_mse
        if epoch_total < best_loss - 1e-6:
            best_loss = epoch_total
            no_improve = 0
            torch.save(model.state_dict(), model_path) # 及时保存最优
            torch.save(model.temp_connector.export_abs(), abs_path)
            logging.info('Best model saved.')
        else:
            no_improve += 1

        scheduler.step(epoch_total)

        if no_improve >= early_stop_patience:
            logging.info('Early stopping.')
            break

    logging.info(f'Training completed.')

def test(model_path, labeled_dir, test_result_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = GMANetPreTrain().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 准备 CSV - 加入 mape 标题
    test_result_path.parent.mkdir(parents=True, exist_ok=True)
    header = ['folder', 'battery_count', 'mse', 'mae', 'rmse', 'mape']
    need_header = not test_result_path.exists()
    
    with open(test_result_path, 'a') as f:
        if need_header:
            f.write(','.join(header) + '\n')

        for subdir in tqdm(sorted(labeled_dir.iterdir()), desc='Testing'):
            if not subdir.is_dir() or not subdir.name.startswith('test'):
                continue

            files = sorted(subdir.glob('*.pkl'))
            battery_count = len(files)
            preds_all = []
            trues_all = []

            for file in files:
                battery = BatteryData.load(str(file))
                qc_list = [np.asarray(c.labeled_Qc, dtype=np.float32) for c in battery.cycle_data]
                soh_list = [float(c.labeled_soh) for c in battery.cycle_data]

                X = torch.tensor(np.stack(qc_list), dtype=torch.float32).to(device)
                with torch.no_grad():
                    yhat = model(X).cpu().squeeze(1).numpy()
                preds_all.extend(yhat.tolist())
                trues_all.extend(soh_list)

            preds_arr = np.asarray(preds_all, dtype=np.float32)
            trues_arr = np.asarray(trues_all, dtype=np.float32)
            
            # --- 指标计算 ---
            mse = float(np.mean((preds_arr - trues_arr) ** 2))
            mae = float(np.mean(np.abs(preds_arr - trues_arr)))
            rmse = float(np.sqrt(mse))
            
            # 计算 MAPE: 注意 SOH 不能为 0（通常电池 SOH > 0.6）
            # 结果以百分比表示，例如 0.01 表示 1%
            mape = float(np.mean(np.abs((trues_arr - preds_arr) / trues_arr)))

            f.write(f"{subdir.name},{battery_count},{mse:.6f},{mae:.6f},{rmse:.6f},{mape:.6f}\n")
    




if __name__ == "__main__":

    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="main")
    config = OmegaConf.create(config)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    labeled_dir = config.path.labeled_dir
    train_dir = os.path.join(labeled_dir, 'train')
    
    results_dir = config.path.pretrain_results_dir
    models_dir = config.path.models_dir
    model_path = os.path.join(models_dir, 'GMA-NET.pkl')
    abs_path = os.path.join(models_dir, 'GMA_pretrained_abs.pth')
    test_result_path = os.path.join(results_dir, 'GMANetPreTrain_test.csv')
    os.makedirs(results_dir, exist_ok=True)

    X, y, samples_weights = read_dataset(Path(train_dir))
    train(X, y, samples_weights, Path(model_path), Path(abs_path))
    test(Path(model_path), Path(labeled_dir), Path(test_result_path))
