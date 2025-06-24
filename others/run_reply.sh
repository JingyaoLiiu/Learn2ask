MODEL=/public/home/ldk/model_cards/Qwen3-8B
DATA_PATH=full_test_data.jsonl
THINKING=false
SNOWBALL=false
OUT_PATH=full_test_data_reply_${THINKING}_${SNOWBALL}.jsonl

python3 process_reply_data.py $MODEL $DATA_PATH $OUT_PATH $THINKING $SNOWBALL

CUDA_VISIBLE_DEVICES=0 python3 run_sampling.py \
                                --model-path $MODEL \
                                --data-path $OUT_PATH \
                                --output-path $OUT_PATH \
                                --temperature 0.6 \
                                --num-outputs 4