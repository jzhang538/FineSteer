#!/usr/bin/env bash
set -euo pipefail

# Define common parameters
MODEL_NAME="llama3"
DS_PATH="./data_tqa/llama3_ans_avg_seed0_testsize0.5_layers_12"
DS_PATH="./data_tqa/llama3_ans_avg_seed0_testsize0.5_layers_12"
LAYERS=12
SEED=0
METHOD="steer"
#OPENGEN_EVAL="--opengen_eval"   # 留空则不会添加该参数
# EVAL_METHOD="gpt"
ALPHA=2
EPOCHS=100
#cluster_modes=("base" "delta_pca" "joint")
# k sweep
# KS=(8 9 11 12 13)
# KS=(8)
K=10
ALPHAS=(2.5 ) 
for ALPHA in "${ALPHAS[@]}"; do
  echo "Starting flow.py with k=${K} and alpha=${ALPHA}"
  python flow.py \
    --model_name "${MODEL_NAME}" \
    --ds_path "${DS_PATH}" \
    --layers "${LAYERS}" \
    --seed "${SEED}" \
    --method "${METHOD}" \
    --cluster_mode "base" \
    --train \
    --k "${K}" \
    --choose_method "top-k" \
    --alpha "${ALPHA}" \
    --gpus 0 \
    --num_epochs "${EPOCHS}"

  echo "Finished flow.py with k=${K}"
  echo "------------------------------------"
done

echo "All scripts finished successfully."
