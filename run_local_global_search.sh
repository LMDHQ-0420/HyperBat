#!/bin/bash
mkdir -p logs/grid_search

# 定义要搜索的参数组合
# 格式: "local_num_cycles global_num_cycles"
combinations=(
    # "5 20"
    "8 20"
    "10 20"
    "12 30"
    "15 30"
    "18 40"
    "20 40"
    "25 50"
    "30 60"
    "35 70"
    "40 80"
    "45 90"
    "50 100"
    "20 80"
    "10 100"
)

for combo in "${combinations[@]}"; do
    local_c=$(echo $combo | cut -d' ' -f1)
    global_c=$(echo $combo | cut -d' ' -f2)
    echo "Running local=$local_c, global=$global_c"
    
    python 05_train_WeightDenoiser.py --window_size 200 --stride 50 --diffusion_steps 1 --local_num_cycles $local_c --global_num_cycles $global_c > logs/grid_search/05_l${local_c}_g${global_c}_s1.txt 2>&1
    if [ $? -ne 0 ]; then
        echo "Training failed for l=$local_c, g=$global_c" >> logs/grid_search/errors.txt
        continue
    fi
    
    python 06_eval_end2end.py --window_size 200 --stride 50 --diffusion_steps 1 --local_num_cycles $local_c --global_num_cycles $global_c > logs/grid_search/06_l${local_c}_g${global_c}_s1.txt 2>&1
    if [ $? -ne 0 ]; then
        echo "Evaluation failed for l=$local_c, g=$global_c" >> logs/grid_search/errors.txt
    fi
done