#!/bin/bash
mkdir -p logs

# 每次脚本启动创建独立日志目录，避免不同批次日志互相覆盖
run_ts=$(date +"%y-%m-%d_%H-%M-%S")
run_log_dir="logs/${run_ts}"
mkdir -p "${run_log_dir}"

# 固定 local/global，同时测试 diff 和 flow
local_cycles=40
global_cycles=200
window_size=200
stride=50

# 按需求：step 取 1 和 40
steps=(1 40)

# 按需求：dim 给两个值（可按需改）
dims=(256 512)

# 同时测试两种 denoiser
types=(diff flow)

error_log="${run_log_dir}/errors_diff_flow_l40_g200.txt"

for dim in "${dims[@]}"; do
    for denoiser_type in "${types[@]}"; do
        for step in "${steps[@]}"; do
            echo "Running type=${denoiser_type}, local=${local_cycles}, global=${global_cycles}, steps=${step}, dim=${dim}"

            train_log="${run_log_dir}/05_${denoiser_type}_l${local_cycles}_g${global_cycles}_s${step}_d${dim}_w${window_size}_s${stride}.txt"
            eval_log="${run_log_dir}/06_${denoiser_type}_l${local_cycles}_g${global_cycles}_s${step}_d${dim}_w${window_size}_s${stride}.txt"

            python 05_train_WeightDenoiser.py \
                --denoiser_type "${denoiser_type}" \
                --window_size "${window_size}" \
                --stride "${stride}" \
                --local_num_cycles "${local_cycles}" \
                --global_num_cycles "${global_cycles}" \
                --diffusion_steps "${step}" \
                --denoiser_hidden_dim "${dim}" \
                > "${train_log}" 2>&1

            if [ $? -ne 0 ]; then
                echo "Training failed: type=${denoiser_type}, l=${local_cycles}, g=${global_cycles}, steps=${step}, dim=${dim}" >> "${error_log}"
                continue
            fi

            python 06_eval_end2end.py \
                --denoiser_type "${denoiser_type}" \
                --window_size "${window_size}" \
                --stride "${stride}" \
                --local_num_cycles "${local_cycles}" \
                --global_num_cycles "${global_cycles}" \
                --diffusion_steps "${step}" \
                --denoiser_hidden_dim "${dim}" \
                > "${eval_log}" 2>&1

            if [ $? -ne 0 ]; then
                echo "Evaluation failed: type=${denoiser_type}, l=${local_cycles}, g=${global_cycles}, steps=${step}, dim=${dim}" >> "${error_log}"
            fi
        done
    done
done
