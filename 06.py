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

from battery_data import BatteryData
from model import GMANet, HyperLoRAGenerator

# ==========================================
# 1. 核心推理逻辑
# ==========================================

def run_detailed_evaluation(base_model, hyper_gen, weights_dir, labeled_dir, device, args, result_path):
    """
    端到端评估：记录每一个电池每一个窗口的 MSE, MAE, RMSE
    """
    detailed_results = []
    
    # 获取测试子目录 (test_CALCE, test_HNEI, etc.)
    test_subdirs = sorted([d for d in weights_dir.iterdir() if d.is_dir() and d.name.startswith('test_')])
    
    if not test_subdirs:
        logging.error(f"在路径 {weights_dir} 下未找到测试文件夹。")
        return

    for subdir in test_subdirs:
        dataset_name = subdir.name
        raw_data_dir = labeled_dir / dataset_name
        weight_files = sorted(subdir.glob('*.pkl'))
        
        if not weight_files:
            continue
        
        logging.info(f"正在深度评估数据集: {dataset_name}")
        
        for f in tqdm(weight_files, desc=f"Eval {dataset_name}"):
            # --- 1. 解析文件名获取元数据 ---
            # 假设文件名格式为: BatteryName_StartIdx.pkl
            parts = f.stem.split("_")
            start_idx = int(parts[-1])
            battery_name = "_".join(parts[:-1])
            battery_path = raw_data_dir / f"{battery_name}.pkl"
            
            if not battery_path.exists():
                continue
                
            battery = BatteryData.load(str(battery_path))
            
            # --- 2. 准备信号数据 ---
            # Global 特征信号
            g_data = battery.cycle_data[0 : args.global_num_cycles]
            g_cyc = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in g_data])).to(device)
            
            # Local 特征信号
            l_data = battery.cycle_data[start_idx : start_idx + args.local_num_cycles]
            l_cyc = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in l_data])).to(device)
            
            # 待预测窗口信号
            w_data = battery.cycle_data[start_idx : start_idx + args.window_size]
            w_cyc = torch.tensor(np.stack([np.asarray(c.labeled_Qc, dtype=np.float32) for c in w_data])).to(device)
            true_soh = np.array([c.labeled_soh for c in w_data])

            # --- 3. 端到端预测 ---
            with torch.no_grad():
                # A. 提取指纹特征
                feat_g = base_model.encoder(g_cyc).unsqueeze(0) # [1, G, 64]
                feat_l = base_model.encoder(l_cyc).unsqueeze(0) # [1, L, 64]
                
                # B. 生成 HyperLoRA 权重
                p_A, p_B, p_log_s = hyper_gen(feat_g, feat_l)
                
                # C. 提取窗口特征
                feat_w = base_model.encoder(w_cyc) # [Window, 64]
                
                # D. 注入权重并计算 SOH
                # 将生成的单组权重扩展到整个 Batch (Window Size)
                p_A_batch = p_A.repeat(feat_w.size(0), 1, 1)
                p_B_batch = p_B.repeat(feat_w.size(0), 1, 1)
                p_log_s_batch = p_log_s.repeat(feat_w.size(0), 1)
                
                pred_soh = base_model.apply_lora(feat_w, p_A_batch, p_B_batch, p_log_s_batch)
                pred_soh = pred_soh.cpu().numpy().flatten()

            # --- 4. 计算多维误差指标 ---
            mse = np.mean((pred_soh - true_soh) ** 2)
            mae = np.mean(np.abs(pred_soh - true_soh))
            rmse = np.sqrt(mse)

            # --- 5. 保存详细数据 ---
            detailed_results.append({
                'dataset': dataset_name,
                'battery_name': battery_name,
                'window_start': start_idx,
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            })

    # --- 6. 导出数据 ---
    df = pd.DataFrame(detailed_results)
    df.to_csv(result_path, index=False)
    
    # 打印简要总结
    summary = df.groupby('dataset')[['mae', 'rmse']].mean()
    logging.info("\n=== 评估结果摘要 ===")
    logging.info(f"\n{summary}")
    logging.info(f"详细评估报告已保存至: {result_path}")

# ==========================================
# 2. Main 脚本入口
# ==========================================

if __name__ == "__main__":
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="main")
    config = OmegaConf.create(config)

    parser = argparse.ArgumentParser(description="HyperBat End-to-End Detailed Evaluation")
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = Path(config.path.results_dir) / f"end2end_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir = Path(config.path.labeled_dir)
    # 权重路径需与 Phase 2 的输出保持一致
    weights_dir = Path(config.path.weights_dir) / f'wsize_{args.window_size}_stride_{args.stride}'

    # 模型路径
    model_path = Path(config.path.results_dir) / 'GMA-NET.pkl'

    if not args.slide:
        weights_dir = Path(config.path.weights_dir) / 'full'
        hyper_model_path = Path(config.path.results_dir) / f'HyperLoRA_full_global{args.global_num_cycles}_local{args.local_num_cycles}.pkl'
        result_path = result_dir / f'End2End_full_global{args.global_num_cycles}_local{args.local_num_cycles}_test.csv'
    else:
        weights_dir = Path(config.path.weights_dir) / f'wsize_{args.window_size}_stride_{args.stride}'
        hyper_model_path = Path(config.path.results_dir) / f'HyperLoRA_wsize{args.window_size}_stride_{args.stride}_global{args.global_num_cycles}_local{args.local_num_cycles}.pkl'
        result_path = result_dir / f'End2End_wsize{args.window_size}_stride_{args.stride}_global{args.global_num_cycles}_local{args.local_num_cycles}.csv'

    # --- 模型加载 ---
    # 1. 加载 GMA-NET (Base)
    base_model = GMANet(rank=4).to(device)
    # 注意: GMA-NET.pkl 可能包含 temp_connector，我们设 strict=False 以兼容
    base_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    # 2. 加载 HyperLoRA 生成器
    hyper_gen = HyperLoRAGenerator(feature_dim=64, rank=4).to(device)
    hyper_gen.load_state_dict(torch.load(hyper_model_path, map_location=device))

    base_model.eval()
    hyper_gen.eval()

    logging.info("--- 开始 HyperBat 端到端深度评估 (逐窗口记录) ---")
    
    run_detailed_evaluation(
        base_model, 
        hyper_gen, 
        weights_dir, 
        labeled_dir, 
        device, 
        args, 
        result_path
    )