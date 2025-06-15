DEVICE=0

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL="/public/home/ldk/model_cards/Qwen3-8B"
DATA_PATH="/public/home/ldk/users/wat/learn2ask/results/Qwen3-8B_test_think_step2.jsonl"
OUTPUT_PATH="/public/home/ldk/users/wat/learn2ask/results/tmp.jsonl"

CUDA_VISIBLE_DEVICES=$DEVICE python3 13_generate_solution_full_traj.py $MODEL $DATA_PATH $OUTPUT_PATH