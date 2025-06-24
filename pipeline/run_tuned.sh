DEVICE=0

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL=/public/home/ldk/users/wat/learn2ask/sft/qwen3_8b_think/global_step_103
DATA_PATH=/public/home/ldk/users/wat/learn2ask/dataset/test.jsonl
OUTPUT_PATH=$MODEL/test_think_step1.jsonl
AGENT_MODEL=/public/home/ldk/model_cards/Qwen3-14B

CUDA_VISIBLE_DEVICES=$DEVICE python3 11_generate_question.py $MODEL $DATA_PATH $OUTPUT_PATH

DATA_PATH=$OUTPUT_PATH
OUTPUT_PATH=$MODEL/test_think_step2.jsonl

CUDA_VISIBLE_DEVICES=$DEVICE python3 12_generate_user_reply.py $AGENT_MODEL $DATA_PATH $OUTPUT_PATH

DATA_PATH=$OUTPUT_PATH
OUTPUT_PATH=$MODEL/test_think_step3.jsonl

CUDA_VISIBLE_DEVICES=$DEVICE python3 13_generate_solution_full_traj.py $MODEL $DATA_PATH $OUTPUT_PATH

DATA_PATH=/public/home/ldk/model_cards/gsm8k
OUTPUT_PATH=$MODEL/test_think_normal.jsonl

CUDA_VISIBLE_DEVICES=$DEVICE python3 13_generate_solution_normal.py $MODEL $DATA_PATH $OUTPUT_PATH