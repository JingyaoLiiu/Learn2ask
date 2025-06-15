export CUDA_HOME=/public/home/ldk/cuda/cuda-12.1

export WANDB_PROJECT=learn2ask-sft-Qwen3-4B-think
export WANDB_DIR=/public/home/ldk/users/ljy/learn2ask/sft/wandb 
export WANDB_NAME=Qwen3-4B-sft-lr1e-6-bsz8

CUDA_VIDIBLE_DEVICES=0,1 WANDB_MODE=offline accelerate launch --config_file="/public/home/ldk/users/ljy/learn2ask/sft/default_config.yaml" /public/home/ldk/users/ljy/learn2ask/sft/train.py \
    --data_fpath /public/home/ldk/users/ljy/learn2ask/sft/data/Qwen3-4B_sft_formated.jsonl \
    --model_fpath /public/home/ldk/model_cards/Qwen3-4B \
    --output_dir /public/home/ldk/users/ljy/learn2ask/sft/model/Qwen3-4B_lr1e-5_bsz32 \
    --per_device_train_batch_size 16 \
    --response_template "assistant" \
    --instruction_template "user" \
    --gradient_checkpointing \
    --num_train_epochs 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --save_strategy epoch \
    --bf16 \
    --logging_first_step \
    --logging_steps 1 \
    --report_to wandb

