DEVICE=1

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL=/public/home/ldk/users/ljy/learn2ask/verl_sft/trained_models/llama3.2-3b-instruct-without-hint/global_step_72
DATA_PATH=/public/home/ldk/model_cards/gsm8k
# OUTPUT_PATH=/public/home/ldk/users/wat/learn2ask/results/Qwen3-1.7B_test_think_normal.jsonl
OUTPUT_PATH=/public/home/ldk/users/ljy/learn2ask/eval/test/llama3.2-3b-instruct-without-hint_test_normal.jsonl
CUDA_VISIBLE_DEVICES=$DEVICE python3 13_generate_solution_normal.py $MODEL $DATA_PATH $OUTPUT_PATH