#!/usr/bin/env bash
set -euo pipefail

# Define common parameters
MODEL_NAME="gemma2"
DS_PATH="./data_tqa/gemma-2_ans_avg_seed0_testsize0.5_layers_18_20_22"
#DS_PATH="./data_tqa/llama-3_ans_avg_seed0_testsize0.5_layers_11_12_13"
LAYERS=20
SEED=0
METHOD="steer"
# OPENGEN_EVAL="--opengen_eval"   # 留空则不会添加该参数
# EVAL_METHOD="gpt"
ALPHA=1.5
EPOCHS=40
# cluster_modes=("base" "delta_pca" "joint")
methods=("truthflow" "base" "alphasteer" "dola" "base")
# k sweep
# KS=(8 9 11 12 13)
# KS=(8)
K=20
for method in "${methods[@]}"; do
  echo "Starting flow.py with k=${K} and method=${method}"
  python flow.py \
    --model_name "${MODEL_NAME}" \
    --ds_path "${DS_PATH}" \
    --layers "${LAYERS}" \
    --seed "${SEED}" \
    --method "${method}" \
    --train \
    --k "${K}" \
    --choose_method "top-k" \
    --alpha "${ALPHA}" \
    --num_epochs "${EPOCHS}"

  echo "Finished flow.py with k=${K}"
  echo "------------------------------------"
done

echo "All scripts finished successfully."
