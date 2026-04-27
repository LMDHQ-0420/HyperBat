#!/bin/bash
mkdir -p logs

# 固定 local/global，仅搜索 diffusion_steps
steps=(
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
)

for step in "${steps[@]}"; do
    echo "Running local=5, global=20, diffusion_steps=$step"

    python 05_train_WeightDenoiser.py --denoiser_type flow --window_size 200 --stride 50 --local_num_cycles 5 --global_num_cycles 20 --diffusion_steps $step > logs/05_flow_l5_g20_s${step}_d512_w200_s50.txt 2>&1
    if [ $? -ne 0 ]; then
        echo "Training failed for l=5, g=20, steps=$step" >> logs/errors_flow.txt
        continue
    fi

    python 06_eval_end2end.py --denoiser_type flow --window_size 200 --stride 50 --local_num_cycles 5 --global_num_cycles 20 --diffusion_steps $step > logs/06_flow_l5_g20_s${step}_d512_w200_s50.txt 2>&1
    if [ $? -ne 0 ]; then
        echo "Evaluation failed for l=5, g=20, steps=$step" >> logs/errors_flow.txt
    fi
done
