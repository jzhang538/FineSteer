#!/usr/bin/env bash
set -euo pipefail

# Define common parameters
MODEL_NAME="llama3.2"
DS_PATH="./data_tqa/llama3.2_ans_avg_seed0_testsize0.5_layers_8_9_10_11_12_13"
#DS_PATH="./data_tqa/llama-3_ans_avg_seed0_testsize0.5_layers_11_12_13"
LAYERS=11
SEED=0
METHOD="base"
OPENGEN_EVAL="--opengen_eval"   # 留空则不会添加该参数
# EVAL_METHOD="gpt"
ALPHAS=2.5
EPOCHS=100

# k sweep
# KS=(8 9 11 12 13)
# KS=(8)
K=8 
for ALPHA in "${ALPHAS[@]}"; do
  echo "Starting flow.py with k=${K}"
  python flow.py \
    --model_name "${MODEL_NAME}" \
    --ds_path "${DS_PATH}" \
    --layers "${LAYERS}" \
    --seed "${SEED}" \
    --method "${METHOD}" \
    ${OPENGEN_EVAL} \
    --train \
    --k "${K}" \
    --choose_method "top-k" \
    --alpha "${ALPHA}" \
    --num_epochs "${EPOCHS}"

  echo "Finished flow.py with k=${K}"
  echo "------------------------------------"
done

echo "All scripts finished successfully."
