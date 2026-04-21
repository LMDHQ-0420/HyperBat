#!/bin/bash
mkdir -p logs/grid_search

# 固定 local/global，仅搜索 diffusion_steps
steps=(
    1
    3
    5
    10
    15
    20
    30
    40
    50
    75
    100
    125
    150
    75
    200
    250
    300
    350
    500
)

for step in "${steps[@]}"; do
    echo "Running local=5, global=20, diffusion_steps=$step"

    python 05_train_WeightDenoiser.py --window_size 200 --stride 50 --local_num_cycles 5 --global_num_cycles 20 --diffusion_steps $step > logs/grid_search/05_l5_g20_s${step}.txt 2>&1
    if [ $? -ne 0 ]; then
        echo "Training failed for l=5, g=20, steps=$step" >> logs/grid_search/errors.txt
        continue
    fi

    python 06_eval_end2end.py --window_size 200 --stride 50 --local_num_cycles 5 --global_num_cycles 20 --diffusion_steps $step > logs/grid_search/06_l5_g20_s${step}.txt 2>&1
    if [ $? -ne 0 ]; then
        echo "Evaluation failed for l=5, g=20, steps=$step" >> logs/grid_search/errors.txt
    fi
done
