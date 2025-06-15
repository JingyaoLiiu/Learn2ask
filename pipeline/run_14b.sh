DEVICE=0

MODEL="/public/home/ldk/model_cards/Qwen3-14B"
DATA_PATH="/public/home/ldk/users/wat/learn2ask/dataset/test.jsonl"
OUTPUT_PATH="/public/home/ldk/users/wat/learn2ask/results/Qwen3-14B_test_think_step1.jsonl"

CUDA_VISIBLE_DEVICES=$DEVICE python3 11_generate_question.py $MODEL $DATA_PATH $OUTPUT_PATH

MODEL="/public/home/ldk/model_cards/Qwen3-14B"
DATA_PATH="/public/home/ldk/users/wat/learn2ask/results/Qwen3-14B_test_think_step1.jsonl"
OUTPUT_PATH="/public/home/ldk/users/wat/learn2ask/results/Qwen3-14B_test_think_step2.jsonl"

CUDA_VISIBLE_DEVICES=$DEVICE python3 12_generate_user_reply.py $MODEL $DATA_PATH $OUTPUT_PATH

MODEL="/public/home/ldk/model_cards/Qwen3-14B"
DATA_PATH="/public/home/ldk/users/wat/learn2ask/results/Qwen3-14B_test_think_step2.jsonl"
OUTPUT_PATH="/public/home/ldk/users/wat/learn2ask/results/Qwen3-14B_test_think_step3.jsonl"

CUDA_VISIBLE_DEVICES=$DEVICE python3 13_generate_solution.py $MODEL $DATA_PATH $OUTPUT_PATH