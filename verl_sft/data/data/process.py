import argparse
import os

import pandas as pd
import jsonlines

from transformers import AutoTokenizer

data_path = "/root/paddlejob/workspace/env_run/lyj/Learn2ask/sft_verl/raw_data/rl_train_raw_rollout16_Meta-Llama-3.1-8B-Instruct.jsonl"


# Read data
with jsonlines.open(data_path, "r") as reader:
    dataset = list(reader)


first_round_prompt = "If the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question within \\boxed{}."

second_round_prompt = "Answer to your question: [ANSWER]\n\nWith these information, answer the previous question."

# Create example conversations
conversations = []
for instance in dataset:
    messages = [
        {
            "role": "user",
            "content": first_round_prompt.replace("[QUESTION]", instance["modified"]),
        },
        {
            "role": "assistant",
            "content": (
                instance["assistant_1"].strip()
                if "thinking" in data_path
                else "<think>\n\n</think>\n\n" + instance["assistant_1"].strip()
            ),
        },
        {
            "role": "user",
            "content": second_round_prompt.replace("[ANSWER]", instance["user_reply"]),
        },
        {
            "role": "assistant",
            "content": (
                instance["solution"].strip()
                if "thinking" in data_path
                else "<think>\n\n</think>\n\n" + instance["solution"].strip()
            ),
        },
    ]
    conversations.append(
        {
            "messages": messages,
            "enable_thinking": True if "thinking" in data_path else False,
        }
    )

# tokenizer = AutoTokenizer.from_pretrained(
#     "/root/paddlejob/workspace/env_run/lyj/rrrl/_hf_resources/model/Qwen/Qwen3-4B"
# )

# tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"


# query = tokenizer.apply_chat_template(
#     conversations[444]["messages"],
#     tokenize=False,
#     add_generation_prompt=False,
#     enable_thinking=True if "thinking" in data_path else False,
# )
# print(query)

# Save to parquet files
train_df = pd.DataFrame(conversations)
train_df.to_parquet(data_path.replace(".jsonl", ".parquet"))
