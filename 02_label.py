import logging
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
import argparse
import os
from tqdm import tqdm
import numpy as np

from battery_data import BatteryData, CycleData


def _fill_nan_interp(arr):
    """Fill NaN/Inf by linear interpolation; if all invalid, return zeros of same shape."""
    arr = np.asarray(arr, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        return arr
    if not finite_mask.any():
        return np.zeros_like(arr)
    x = np.arange(arr.size)
    arr[~finite_mask] = np.interp(x[~finite_mask], x[finite_mask], arr[finite_mask])
    return arr

PUBLIC_DATASETS = (
    'MATR',
    'HUST',
    'CALCE',
    'RWTH',
    'HNEI',
    # 'SNL',
    # 'UL-PUR',
    'CALB',
    'ISU-ILCC',
    'MICH',
    'MICH_EXP',
    'NAion',
    'SDU',
    'Stanford',
    'Tongji',
    'XJTU',
    'ZNion'
)

# 计算 SOH, 剔除SOH小于70的cycle
def get_soh(battery: BatteryData) -> BatteryData:
    base_cap = 0.0
    for cycle in battery.cycle_data[:5]:
        base_cap += max(cycle.discharge_capacity_in_Ah)
    base_cap /= 5.0

    for cycle in battery.cycle_data:
        Qd = max(cycle.discharge_capacity_in_Ah)
        cycle.labeled_soh = Qd / base_cap
    battery.cycle_data = [cycle for cycle in battery.cycle_data if cycle.labeled_soh >= 0.7 and cycle.labeled_soh <= 1.0]
    return battery


def voltage_grid_resample(
    v,
    i,
    Qc,
    start_V,
    end_V,
    dv=1e-3,
):

    v = np.asarray(v, dtype=np.float64)
    i = np.asarray(i, dtype=np.float64)
    Qc = np.asarray(Qc, dtype=np.float64)

    # 保证原始数据电压是单调递增的，以便 np.interp 正确工作
    if v[0] > v[-1]:
        idx = np.argsort(v)
        v = v[idx]
        i = i[idx]
        Qc = Qc[idx]

    # 1. 构造固定的电压网格 [start_V, end_V]
    # 使用 np.arange 时，为了包含 end_V，建议增加一个微小的 offset
    v_new = np.arange(start_V, end_V + 0.5 * dv, dv)

    # 2. 在“电压坐标系”上进行线性插值
    # left=0, right=0 表示当 v_new 的点超出 v 的范围时，填充 0
    i_new = np.interp(v_new, v, i, left=0.0, right=0.0)
    Qc_new = np.interp(v_new, v, Qc, left=0.0, right=0.0)

    return v_new[:-1], i_new[:-1], Qc_new[:-1]



# 截取充电曲线电压窗口
def get_labeled_qc(battery: BatteryData, dataset: str) -> BatteryData:

    V_window_range = {
        # 标准材料
        'LCO': (3.5, 3.9),
        'NMC': (3.5, 3.9),
        'NCA': (3.5, 3.9),
        'LFP': (3.1, 3.5),
        'ZNion': (1.1, 1.5),                     # 锌离子电池，典型电压平台较低
        'NAion': (2.7, 3.1),                   # 钠电典型电压窗口
        'NMC+NCA': (3.5, 3.9)
    }
    
    delete_indices = []
    for cycle in battery.cycle_data:
        V = cycle.voltage_in_V or []
        I = cycle.current_in_A or []
        t = cycle.time_in_s or []
        Qc = cycle.charge_capacity_in_Ah

        # 截取充电曲线
        cut_t = len(I)
        for idx in range(5, len(I) - 5):
            if I[idx-1] * I[idx+1] < 0:
                cut_t = idx
                break

        if dataset == 'RWTH':   # RWTH先放电后充电
            charge_V = V[cut_t:]
            charge_I = I[cut_t:]
            charge_t = t[cut_t:]
            charge_Qc = Qc[cut_t:]
        else:
            charge_V = V[:cut_t]
            charge_I = I[:cut_t]
            charge_t = t[:cut_t]
            charge_Qc = Qc[:cut_t]

        # 截取电压窗口
        (start_V, end_V) = V_window_range.get(battery.cathode_material)
        start_t, end_t = None, None
        prev_v = None
        for idx, val in enumerate(charge_V):
            if start_t is None and val >= start_V:
                start_t = idx
            # 提前结束：电压开始下降
            if prev_v is not None and val < prev_v:
                end_t = idx
                break
            if val >= end_V:
                end_t = idx
                break
            prev_v = val
        start_t = start_t or 0
        end_t = end_t or len(charge_V)
        window_V = _fill_nan_interp(charge_V[start_t:end_t])
        window_I = _fill_nan_interp(charge_I[start_t:end_t])
        window_Qc = _fill_nan_interp(charge_Qc[start_t:end_t])
        if len(window_V) == 0:  # 删除空窗口
            delete_indices.append(cycle.cycle_number)
            continue
        if max(window_V) - min(window_V) < (end_V - start_V) * 0.5:  # 删除电压范围过小的cycle
            delete_indices.append(cycle.cycle_number)
            continue
        if np.mean(window_V) < start_V or np.mean(window_V) > end_V:    # 删除不在窗口范围内的电压
            delete_indices.append(cycle.cycle_number)
            continue
        
        # 电压网格化重采样
        cycle.labeled_V, cycle.labeled_I, cycle.labeled_Qc = voltage_grid_resample(
            window_V, window_I, window_Qc, start_V, end_V
        )
    
    # 执行删除
    battery.cycle_data = [cycle for cycle in battery.cycle_data if cycle.cycle_number not in delete_indices]
    return battery


def run_label(preprocessed_dir, train_dir, test_dir, train_ratio, dataset):
    file_num = len(list(preprocessed_dir.glob("*.pkl")))
    train_num = int(file_num * train_ratio)

    skip = []
    index = 0
    for file in tqdm(sorted(preprocessed_dir.glob("*.pkl")), desc=f'Labeling {preprocessed_dir}'):
        battery = BatteryData.load(str(file))
        battery = get_soh(battery)
        battery = get_labeled_qc(battery, dataset)
        if len(battery.cycle_data) > 200:
            if index < train_num:
                output_file = train_dir / file.name
            else:
                output_file = test_dir / file.name
            battery.dump(str(output_file))
        else:
            skip.append(file.name)
        index += 1
    logging.info(f'Labeling completed for dataset: {dataset}.')
    if skip:
        logging.warning(f'Skipped files due to insufficient valid cycles after labeling: {skip}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[*PUBLIC_DATASETS, 'all'],
        default='all',
        help="Dataset to label"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="main")
    config = OmegaConf.create(config)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    datasets_to_label = PUBLIC_DATASETS if args.dataset == 'all' else [args.dataset]
    for dataset in datasets_to_label:
        preprocessed_dir = Path(config.path.preprocessed_dir) / dataset
        train_dir = Path(config.path.labeled_dir) / 'train'
        test_dir = Path(config.path.labeled_dir) / f'test_{dataset}'
        train_ratio = config.train_ratio.get(dataset)

        os.makedirs(train_dir, exist_ok=True)
        if train_ratio < 1.0:
            os.makedirs(test_dir, exist_ok=True)

        run_label(preprocessed_dir, train_dir, test_dir, train_ratio, dataset)