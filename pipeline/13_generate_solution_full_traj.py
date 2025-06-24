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

first_round_prompt = "Question: [QUESTION]\n\nIf the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question inside \\boxed{}. If no further question can be asked, respond with \\boxed{None}."

second_round_prompt = "Answer to your question: [ANSWER]\n\nWith these information, answer the previous question."


tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokenizer.chat_template = """
{%- if messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- if '</think>' in content %}
            {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n'}}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n<think>\n\n</think>\n\n' + content + '<|im_end|>' + '\n'}}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
""".strip()


with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

queries = []
anchors = []
for instance in dataset:
    for rollout in instance["rollout"]:
        if "user_reply" in rollout:
            user_1 = first_round_prompt.replace("[QUESTION]", instance["modified"])
            assistant_1 = rollout["r1_reply"]
            user_2 = second_round_prompt.replace("[ANSWER]", rollout["user_reply"])
            query = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": user_1}, {"role": "assistant", "content": assistant_1}, {"role": "user", "content": user_2}], tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
            queries.append(query)
            anchors.append(rollout)

print((queries[0],))

llm = LLM(
            model=MODEL,
            swap_space=8,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=8192 + 1024,
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
    anchor["r2_reply"] = result.outputs[0].text

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(dataset)
