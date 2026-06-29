#!/usr/bin/env bash
set -euo pipefail

# ====== 通用参数 ======
MODEL_NAME="qwen2.5"
DS_PATH="./data_tqa/qwen2.5_ans_avg_seed0_testsize0.5_layers_10_11_12_13_14_15_16_17_18_19_20"
SEED=0
METHOD="base"
OPENGEN_EVAL="--opengen_eval"   # 留空则不会添加该参数
EPOCHS=30

# ====== sweep 范围 ======
LAYERS=(12)
K=10
ALPHAS=(2.5)
methods=("truthflow" "base" "alphasteer" "dola")
for LAYER in "${LAYERS[@]}"; do
  for method in "${methods[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
      echo "Starting flow.py with layer=${LAYER}, k=${K}, alpha=${ALPHA}"
      python flow.py \
        --model_name "${MODEL_NAME}" \
        --ds_path "${DS_PATH}" \
        --layers "${LAYER}" \
        --seed "${SEED}" \
        --method "${method}" \
        --train \
        --k "${K}" \
        --alpha "${ALPHA}" \
        --num_epochs "${EPOCHS}"

      echo "Finished flow.py with layer=${LAYER}, k=${K}, alpha=${ALPHA}"
      echo "------------------------------------"
    done
  done
done

echo "All scripts finished successfully."
