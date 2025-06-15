model_path=/public/home/ldk/model_cards/Qwen3-14B

CUDA_VISIBLE_DEVICES=0,3 vllm serve $model_path --tensor-parallel-size 2