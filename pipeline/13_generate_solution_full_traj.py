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
ENABLE_THINKING = True

first_round_prompt = "If the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question within \\boxed{}."

second_round_prompt = "Answer to your question: [ANSWER]\n\nWith these information, answer the previous question."


tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"


with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)
'''
{
'rollout': [
    {'r1_reply': 'round 1 model response','user_reply':'user reply'},
    {'r1_reply': '','user_reply':''}
 ],
 ...
}
'''
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
            tensor_parallel_size=2,
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
'''
{
'rollout': [
    {'r1_reply': 'round 1 model response','user_reply':'user reply','r2_reply': 'round 2 model response'},
    {'r1_reply': '','user_reply':''}
 ],
 ...
}
'''