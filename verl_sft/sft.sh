#!/bin/bash
set -x

export CUDA_HOME=/public/home/ldk/cuda/cuda-11.7

name=qwen3-1.7b-nothink-without-hint
save_path=./trained_models/$name
model=/public/home/ldk/model_cards/Qwen3-1.7B 
train_file=/public/home/ldk/users/ljy/learn2ask/verl_sft/data/train_without_hint_nothink.parquet # ./data/rl_train_raw_rollout32_8b.parquet
val_file=$train_file

epoch=1 

export CUDA_VISIBLE_DEVICES=0,1  
ulysses_sequence_parallel_size=2
nproc_per_node=2

torchrun --master-port=29600 --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_file \
    data.val_files=$val_file \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=64 \
    data.max_length=4096 \
    optim.lr=1e-5 \
    optim.lr_scheduler=cosine \
    model.partial_pretrain=$model \
    trainer.default_local_dir=$save_path \
    trainer.project_name=sft-l2a \
    trainer.experiment_name=$name \
    trainer.logger=['console'] \
    trainer.total_epochs=$epoch \
    ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    use_remove_padding=true

