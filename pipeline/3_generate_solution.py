import sys
import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# MODEL = "/public/home/ldk/model_cards/Qwen3-1.7B"
# DATA_PATH = "/public/home/ldk/users/wat/learn2ask/dataset/test_clean_data_step2.jsonl"
# OUTPUT_PATH = "/public/home/ldk/users/wat/learn2ask/dataset/test_clean_data_step3.jsonl"
MODEL = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
ENABLE_THINKING = False

first_round_prompt = "Question: [QUESTION]\n\nIf the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question within \\boxed{}."

second_round_prompt = "Answer to your question: [ANSWER]\n\nWith these information, answer the previous question."

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

queries = []
anchors = []
for instance in dataset:
    for rollout in instance["rollouts"]:
        user_1 = first_round_prompt.replace("[QUESTION]", instance["modified"])
        assistant_1 = rollout["question"]
        user_2 = second_round_prompt.replace("[ANSWER]", rollout["answer"])
        query = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": user_1}, {"role": "assistant", "content": assistant_1}, {"role": "user", "content": user_2}], tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
        queries.append(query)
        anchors.append(rollout)

print((queries[0],))

llm = LLM(
            model=MODEL,
            swap_space=8,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=4096 + 1024,
            gpu_memory_utilization=0.8,
            trust_remote_code=True
        )

sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0,
            max_tokens=4096,
            n=1,
            skip_special_tokens=False,
            include_stop_str_in_output=False
        )
results = llm.generate(queries, sampling_params, use_tqdm=True)

for anchor, result in zip(anchors, results):
    anchor["solution"] = [output.text for output in result.outputs]

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(dataset)
