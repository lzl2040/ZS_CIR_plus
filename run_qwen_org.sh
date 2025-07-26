RUN=/mnt/input_zuo/ZS-CIR/plus_version/saves/qwen-4bit-org # 实验名称
# RUN=phi3-4bit-cot # 实验名称
# RUN=qwen-4bit-ke
args=()

# BASE_MODEL="microsoft/Phi-3-vision-128k-instruct"
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

TEMPLATE='*sent_0*\nSummary_above_sentence_in_one_word:'
# TEMPLATE='After_thinking_step_by_step_this_sentence_*sent_0*\nSummary_above_sentence_in_one_word:'
# TEMPLATE="The_essence_of_a_sentence_is_often_captured_by_its_main_subjects_and_actions_while_descriptive_terms_provide_additional_but_less_central_details_With_this_in_mind_this_sentence_*sent_0*\nSummary_above_sentence_in_one_word:"

BIT=4

R=64
ALPHA=16
BATCH_SIZE=768
MICRO_BATCH_SIZE=384 # 尽量调大，直到显存占满, 96 for phi3
EPOCH=2
LR=4e-4 # 4e-4 for llava, 2e-4 for phi3, 4e-4 for qwen

echo $BASE_MODEL
echo $TEMPLATE


echo $MICRO_BATCH_SIZE $BATCH_SIZE

GPUS=1 # 8
NUM_NODES=1 # 4

wandb online


NCCL_DEBUG=ERROR deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES ft_qwen.py \
        --base_model   $BASE_MODEL \
        --data_path 'data/nli_for_simcse.csv' \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE  \
        --num_epochs $EPOCH \
        --learning_rate $LR \
        --cutoff_len 32 \
        --lora_r $R \
        --lora_alpha $ALPHA \
        --lora_dropout 0.05 \
        --output_dir $RUN  --is_sentemb \
        --mask_embedding_sentence_template $TEMPLATE --use_neg_sentence --save_steps 10 \
        --deepspeed ds.config \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj  --logging_steps 1 --grad_checkpoint \
         --load_kbit $BIT \
         ${args[@]}

