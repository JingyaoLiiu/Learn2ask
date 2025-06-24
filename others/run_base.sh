MODEL=/public/home/ldk/model_cards/Qwen3-8B
DATA_PATH=full_test_data.jsonl
THINKING=false
OUT_PATH=full_test_data_base_${THINKING}.jsonl

python3 process_base_data.py $MODEL $DATA_PATH $OUT_PATH $THINKING

CUDA_VISIBLE_DEVICES=1 python3 run_sampling.py \
                                --model-path $MODEL \
                                --data-path $OUT_PATH \
                                --output-path $OUT_PATH \
                                --temperature 0.6 \
                                --num-outputs 4