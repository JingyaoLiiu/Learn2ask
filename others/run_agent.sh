MODEL=/public/home/ldk/model_cards/Qwen3-8B
DATA_PATH=full_test_data.jsonl

OUT_PATH=full_test_data_agent.jsonl

python3 process_agent_data.py $MODEL $DATA_PATH $OUT_PATH

CUDA_VISIBLE_DEVICES=1 python3 run_sampling.py \
                                --model-path $MODEL \
                                --data-path $OUT_PATH \
                                --output-path $OUT_PATH \
                                --temperature 0 \
                                --num-outputs 1