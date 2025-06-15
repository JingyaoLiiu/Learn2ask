set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_HOME=/public/home/ldk/cuda/cuda-12.1
math_train_path=/public/home/ldk/users/wat/learn2ask/utils/train.parquet
math_test_path=/public/home/ldk/users/wat/learn2ask/utils/test.parquet
model_path=/public/home/ldk/model_cards/Qwen3-1.7B
reward_fn_path=/public/home/ldk/users/wat/learn2ask/utils/verl_math_verify.py
n_gpus_per_node=2

train_files="['$math_train_path']"
test_files="['$math_test_path']"

CUDA_VISIBLE_DEVICES=1,2,3 WANDB_MODE=offline python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    custom_reward_function.path=$reward_fn_path \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='learn2ask' \
    trainer.experiment_name='qwen3_4b' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.default_local_dir=/public/home/ldk/users/ljy/learn2ask/rl_model/Qwen3-1.7B_lr5e-7 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
