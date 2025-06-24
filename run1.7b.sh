
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_HOME=/public/home/ldk/cuda/cuda-12.1 
export USER_AGENT=http://12.12.12.5:2580    

math_train_path=/public/home/ldk/users/ljy/l2a/rl/data/rl_train_raw.parquet
math_test_path=/public/home/ldk/users/ljy/l2a/rl/data/rl_test_raw_1shot.parquet 

model_path=/public/home/ldk/users/wat/learn2ask/sft/qwen3_1.7b_think/global_step_103
name=qwen3_1.7b-think

train_files="['$math_train_path']"
test_files="['$math_test_path']"
reward_fn_path=/public/home/ldk/users/ljy/learn2ask/utils/verl_math_verify.py
n_gpus_per_node=2


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.learn2ask=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$reward_fn_path \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='learn2ask' \
    trainer.experiment_name=$name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.rollout_data_dir=./outputs/$name/rollout_data \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.resume_from_path=/public/home/ldk/users/wat/learn2ask/checkpoints/learn2ask/qwen3_1.7b-think/global_step_80 \
    trainer.total_epochs=10 $@