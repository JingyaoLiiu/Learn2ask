import sys
import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset

MODEL = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
ENABLE_THINKING = False

prompt = "Question: [QUESTION]\n\nIf the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question inside \\boxed{}. If no further question can be asked, respond with \\boxed{None}."

tokenizer = AutoTokenizer.from_pretrained(MODEL)

dataset = list(load_dataset(DATA_PATH, 'main', split="test"))

print(len(dataset))

queries = []
anchors = []
for instance in dataset:
    query = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": prompt.replace("[QUESTION]", instance["question"])},], tokenize=False, add_generation_prompt=True)  #enable_thinking=ENABLE_THINKING
    queries.append(query)
    anchors.append(instance)

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
    anchor["rollout"] = [{"r1_reply": output.text} for output in result.outputs]

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(dataset)
