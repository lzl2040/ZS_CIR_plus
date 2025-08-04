#!/bin/bash

# 你的 checkpoint ID 列表
CKPT_LIST=(
  checkpoint-40
)

# 其他参数
BASE_PATH="/mnt/input_zuo/ZS-CIR/plus_version/saves/qwen-4bit-ke"
# BASE_PATH="/home/v-zuoleili/Temp/ZS_CIR_plus/phi3-4bit-ke"
PROMPT_TYPES=("ke" "cot" "org")

# 遍历 checkpoint
for CKPT_ID in "${CKPT_LIST[@]}"; do
  WEIGHT_ROOT="$BASE_PATH/$CKPT_ID"
  echo "Testing checkpoint: $WEIGHT_ROOT"

  # 遍历 prompt_type
  for PROMPT in "${PROMPT_TYPES[@]}"; do
    echo "  Using prompt_type: $PROMPT"

    CUDA_VISIBLE_DEVICES=0 python src/sft_qwen_embed.py \
      --name 'qwen2_5_vl' \
      --base_model 'qwen2_5_vl' \
      --prompt_type "$PROMPT" \
      --file_path '/mnt/input_zuo/ZS-CIR/plus_version/results_wo_ft'
  done
done
