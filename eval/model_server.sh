# source /root/paddlejob/workspace/env_run/lyj/rrrl/eval/.venv/eval/bin/activate

# cd /root/paddlejob/workspace/env_run/lyj/rrrl/_hf_resources/model

model_path=/public/home/ldk/users/ljy/l2a/sft/trained_models/qwen3-1.7b-nothink-with-hint/global_step_120
port=8989


export CUDA_VISIBLE_DEVICES=1
vllm serve $model_path --tensor-parallel-size 1 --port $port 