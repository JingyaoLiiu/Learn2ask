import argparse
import json
import os
import jsonlines
import random
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import concurrent.futures
import multiprocessing
import atexit
from math_verify import parse, verify
from math_verify.errors import TimeoutException
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from transformers import AutoTokenizer

from search_utils import *

MODEL_BASE = "/public/home/ldk/users/ljy/learn2ask/verl_sft/trained_models/llama3.2-3b-instruct-with-hint/global_step_91"  # 放模型的地址， 如：MODEL_BASE/Qwen/Qwen3-14B

parser = argparse.ArgumentParser(description="")

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--think", action="store_true")
parser.add_argument("--noise", action="store_true")
parser.add_argument("--port", type=str, default="8099")
args = parser.parse_args()

model = args.model  # vllm服务的model名称
THINKING = (
    "thinking" in args.model or args.think
)  # thinking 第一轮think，第二次不会think
print("Model:", model)
print("Thinking:", THINKING)


client = OpenAI(
    api_key="sss",
    base_url=f"http://localhost:{args.port}/v1",
)


# 可以换强的模型
client_user = OpenAI(
    api_key="sss",
    base_url="http://12.12.12.5:2580/v1",
)
model_user = "Qwen3-14B"


if not os.path.exists(model):
    model_path = os.path.join(MODEL_BASE, model)
else:
    model_path = model

print(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if "qwen" in model_path.lower():
    CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    tokenizer.chat_template = CHAT_TEMPLATE

#model = "Qwen3-1.7B"

ROLLOUT_TIMES = 1
if not args.noise:
    DATA_PATH = (
        "/public/home/ldk/users/ljy/learn2ask/eval/data/test_raw.jsonl"
    )
else:
    DATA_PATH = (
        "/public/home/ldk/users/ljy/learn2ask/eval/data/test_raw_noise.jsonl"
    )


DIR = "/public/home/ldk/users/ljy/l2a/eval/test/output"
sub_dir = f"{'.'.join(model.split('/')[-2:])}" + ("-thinking" if THINKING else "")

OUTPUT_DIR = os.path.join(DIR, sub_dir)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, os.path.basename(DATA_PATH).replace(".jsonl", "_rollout.jsonl")
)
RESULT_PATH = OUTPUT_PATH.replace(".jsonl", "_result.json")


DEBUG = False


def is_q(hyp):
    if (
        hyp != "None"
        and len(hyp.split()) >= 4
        and (
            "?" in hyp[-5:]
            or hyp.startswith("We need")
            or hyp.startswith("Please provide")
            or hyp.startswith("How")
            or hyp.startswith("What")
            or hyp.startswith("Please specify")
        )
    ):
        return True
    return False


# 读取并采样数据集
with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)
print("Raw data size", len(dataset))

for index, x in enumerate(dataset):
    x["idx"] = index

dataset = dataset * ROLLOUT_TIMES
print("After rollout times", len(dataset))


# 全局进程池（惰性初始化）
_executor = None


def get_executor():
    global _executor
    if _executor is None and multiprocessing.current_process().name == "MainProcess":
        _executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=multiprocessing.cpu_count()
        )
    return _executor


def shutdown_executor():
    global _executor
    if _executor is not None:
        _executor.shutdown()
        _executor = None


atexit.register(shutdown_executor)


def _compute_score_impl(solution_str: str, ground_truth: str) -> float:
    """实际计算分数的核心函数（在子进程中执行）"""
    gold_extraction_config = (LatexExtractionConfig(),)
    pred_extraction_config = (
        LatexExtractionConfig(boxed_match_priority=0),
        ExprExtractionConfig(),
    )

    ret_score = 0.0
    try:
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        gold = parse(ground_truth_boxed, gold_extraction_config)
        pred = parse(solution_str, pred_extraction_config)
        ret_score = verify(gold, pred)
        if not ret_score:
            ground_truth_boxed = "\\boxed{" + ground_truth + "%}"
            gold = parse(ground_truth_boxed, gold_extraction_config)
            ret_score = verify(gold, pred)
    except Exception:
        pass  # 忽略常规异常
    except TimeoutException:
        pass  # 超时返回0.0
    return ret_score


def compute_score(solution_str: str, ground_truth: str) -> float:
    """线程安全的分数计算入口"""
    executor = get_executor()
    if executor is None:
        # 已在子进程：直接计算
        return _compute_score_impl(solution_str, ground_truth)
    else:
        # 主进程：提交到进程池
        future = executor.submit(_compute_score_impl, solution_str, ground_truth)
        return future.result()


# 初始化锁用于文件写入和进度条更新
file_lock = Lock()
pbar = tqdm(total=len(dataset), desc="Processing samples")
pbar_lock = Lock()


def process_sample(instance):

    origin = instance["question"]
    modified = instance["modified"]
    ref = instance["answer"].split("####")[1].strip()

    # 1st assistant
    try:
        if DEBUG:
            print(
                "## Example:",
                [
                    {
                        "role": "user",
                        "content": init_prompt.replace("[QUESTION]", modified),
                    }
                ],
            )
        response = client.completions.create(
            model=model,
            prompt=tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": init_prompt.replace("[QUESTION]", modified),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=THINKING,
            ),
            temperature=1.0,
            max_tokens=4096,
        )
        assistant_1 = response.choices[0].text

        # response = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": init_prompt.replace("[QUESTION]", modified),
        #         }
        #     ],
        #     temperature=1.0,
        #     max_tokens=4096,
        #     # timeout=60,
        #     extra_body={
        #         "chat_template_kwargs": {
        #             "enable_thinking": False if not THINKING else True
        #         }
        #     },
        # )
        # assistant_1 = response.choices[0].message.content

    except Exception as e:
        raise e
    instance["assistant_1"] = assistant_1
    question = extract_text(extract_boxed(assistant_1)).strip()
    has_q = False
    if question is not None and question:
        if is_q(question):
            has_q = True
            instance["assistant_1_q"] = question
    if has_q:
        # 2nd user
        response = None
        retry_times = 8
        for _ in range(retry_times):
            try:
                if DEBUG:
                    print(
                        "## Example:",
                        [
                            {
                                "role": "user",
                                "content": agent_prompt.replace(
                                    "[Context]", origin
                                ).replace("[Question]", question),
                            }
                        ],
                    )
                response = client_user.chat.completions.create(
                    model=model_user,
                    messages=[
                        {
                            "role": "user",
                            "content": agent_prompt.replace(
                                "[Context]", origin
                            ).replace("[Question]", question),
                        }
                    ],
                    temperature=1.0,
                    max_tokens=256,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                break
            except Exception as e:
                print(e)
                continue

        if response is None:
            raise RuntimeError("Failed to get the answer.")

        user_reply = response.choices[0].message.content
        instance["user_reply"] = user_reply

        if DEBUG:
            print("## Answer:", user_reply)
        try:
            if DEBUG:
                print(
                    "## Example:",
                    [
                        {
                            "role": "user",
                            "content": init_prompt.replace("[QUESTION]", modified),
                        },
                        {"role": "assistant", "content": question},
                        {
                            "role": "user",
                            "content": final_prompt.replace("[ANSWER]", user_reply),
                        },
                    ],
                )
            # print(prompt)
            response = client.completions.create(
                model=model,
                prompt=tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": init_prompt.replace("[QUESTION]", modified),
                        },
                        {"role": "assistant", "content": assistant_1},
                        {
                            "role": "user",
                            "content": final_prompt.replace(
                                "[ANSWER]",
                                user_reply if user_reply is not None else "None",
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                temperature=1.0,
                max_tokens=4096,
            )
            solution = response.choices[0].text

            # response = client.chat.completions.create(
            #     model=model,
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": init_prompt.replace("[QUESTION]", modified),
            #         },
            #         {"role": "assistant", "content": assistant_1},
            #         {
            #             "role": "user",
            #             "content": final_prompt.replace("[ANSWER]", user_reply),
            #         },
            #     ],
            #     temperature=1.0,
            #     max_tokens=4096,
            #     # timeout=60,
            #     extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            # )
            # solution = response.choices[0].message.content
        except Exception as e:
            raise e

        corr = compute_score(solution, ref)
        if DEBUG:
            print("## Solution:", (solution,), ref, corr)
        instance["solution"] = solution
        instance["correct"] = corr

    # 保存处理结果
    with file_lock:
        with jsonlines.open(OUTPUT_PATH, mode="a") as writer:
            writer.write(instance)

    # 更新进度条
    with pbar_lock:
        pbar.update(1)

    return instance


# 使用线程池并发处理
start_time = time.time()

with ThreadPoolExecutor(max_workers=64) as executor:
    # 提交所有任务
    futures = [executor.submit(process_sample, sample) for sample in dataset]

    # 等待所有任务完成（结果已实时保存）
    for future in futures:
        future.result()

pbar.close()
total_time = time.time() - start_time
print(f"\nAll samples processed in {total_time:.2f} seconds")
print(f"Results saved to: {OUTPUT_PATH}")

# ============================================
# METRIC
with jsonlines.open(OUTPUT_PATH, mode="r") as reader:
    rollouts = list(reader)

REQ = len([x for x in rollouts if "assistant_1_q" in x]) / len(rollouts)
ACC = len([x for x in rollouts if "correct" in x and x["correct"]]) / len(rollouts)

print(f"ACC: {ACC*100:.4f}, REQ: {REQ*100:.4f}")

with open(RESULT_PATH, "w") as f:
    json.dump({"REQ": REQ, "ACC": ACC}, f)
