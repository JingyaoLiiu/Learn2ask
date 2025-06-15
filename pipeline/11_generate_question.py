import sys
import jsonlines
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# MODEL = "/public/home/ldk/model_cards/Qwen3-1.7B"
# DATA_PATH = "/public/home/ldk/users/wat/learn2ask/dataset/full_test_data_filtered.jsonl"
# OUTPUT_PATH = "/public/home/ldk/users/wat/learn2ask/dataset/test_clean_data_step1.jsonl"

MODEL = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
PROMPT_TYPE = "critic"
ENABLE_THINKING = True

prompt_dict = {
    "base": "Please reason step by step, and put your final answer within \\boxed{}.", # abandon
    "critic": "If the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question within \\boxed{}.",
    "final": "If the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question inside \\boxed{}. If no further question can be asked, respond with \\boxed{None}." # abandon
}

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

# random.seed(42)
# sample_size = int(len(dataset) * 0.25)
# dataset = random.sample(dataset, sample_size)

queries = []
for instance in dataset:
    # for modified question
    query = "Question: " + instance["modified"] + "\n\n" + prompt_dict[PROMPT_TYPE]
    # for original question
    # query = "Question: " + instance["origin"] + "\n\n" + prompt_dict[PROMPT_TYPE]
    query = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": query},], tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
    queries.append(query)

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
            n=1, # sampling num
            skip_special_tokens=False,
            include_stop_str_in_output=False
        )
results = llm.generate(queries, sampling_params, use_tqdm=True)

for instance, result in zip(dataset, results):
    instance["rollout"] = [{"r1_reply": output.text} for output in result.outputs]

print(dataset[0])
print("Save", len(dataset))

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(dataset)

'''
{
 # add rollout
'rollout': [
    {'r1_reply': 'round 1 model response'},
    {'r1_reply': ''}
 ]
}
'''