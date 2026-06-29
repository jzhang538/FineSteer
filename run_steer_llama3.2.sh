#!/usr/bin/env bash
set -euo pipefail

# Define common parameters
MODEL_NAME="llama3.2"
DS_PATH="./data_tqa/llama3.2_ans_avg_seed0_testsize0.5_layers_8_9_10_11_12_13"
LAYERS=11
SEED=0
METHOD="steer"
# OPENGEN_EVAL="--opengen_eval"   # 留空则不会添加该参数
# EVAL_METHOD="gpt"
ALPHA=2
EPOCHS=100
cluster_modes=("base")
# k sweep
# KS=(8 9 11 12 13)
# KS=(8)
K=8 
for cluster_mode in "${cluster_modes[@]}"; do
  echo "Starting flow.py with k=${K} and cluster_mode=${cluster_mode}"
  python flow.py \
    --model_name "${MODEL_NAME}" \
    --ds_path "${DS_PATH}" \
    --layers "${LAYERS}" \
    --seed "${SEED}" \
    --method "${METHOD}" \
    --cluster_mode "${cluster_mode}" \
    --train \
    --k "${K}" \
    --choose_method "top-k" \
    --alpha "${ALPHA}" \
    --gpus 1 \
    --num_epochs "${EPOCHS}"

  echo "Finished flow.py with k=${K}"
  echo "------------------------------------"
done

echo "All scripts finished successfully."
