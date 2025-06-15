MODEL=/public/home/ldk/model_cards/Qwen3-8B
DATA_PATH=full_test_data.jsonl

PROMPT_TYPE=critic
ENABLE_THINKING=true

OUT_PATH=full_test_data_${PROMPT_TYPE}_${ENABLE_THINKING}.jsonl

python3 process_query_data.py $MODEL $DATA_PATH $PROMPT_TYPE $ENABLE_THINKING $OUT_PATH

CUDA_VISIBLE_DEVICES=1 python3 run_sampling.py \
                                --model-path $MODEL \
                                --data-path $OUT_PATH \
                                --output-path $OUT_PATH \
                                --temperature 0.6 \
                                --num-outputs 4