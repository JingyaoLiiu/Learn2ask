import jsonlines
import random
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
from search_utils import *
from transformers import AutoTokenizer
import concurrent.futures
import multiprocessing
import atexit
from math_verify import parse, verify
from math_verify.errors import TimeoutException
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

DATA_PATH = "/public/home/ldk/users/ljy/learn2ask/datasets/rl_train_raw.jsonl"
OUTPUT_PATH = "/public/home/ldk/users/ljy/learn2ask/verl_sft/data/Llama-3.1-8B-Instruct_train_3k_rollout_learn2ask.jsonl"
MODEL = "/public/home/ldk/model_cards/Llama-3.1-8B-Instruct"
DEBUG = False
THINK = False

# ��ʼ��API�ͻ���
client = OpenAI(api_key="universe", base_url="http://localhost:8000/v1")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# tokenizer.chat_template = """
# {%- if messages[0].role == 'system' %}
#     {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
# {%- endif %}
# {%- for message in messages %}
#     {%- if message.content is string %}
#         {%- set content = message.content %}
#     {%- else %}
#         {%- set content = '' %}
#     {%- endif %}
#     {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
#         {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
#     {%- elif message.role == "assistant" %}
#         {%- if '</think>' in content %}
#             {{- '<|im_start|>' + message.role + '\n' + content }}
#         {%- else %}
#             {{- '<|im_start|>' + message.role + '\n<think>\n\n</think>\n\n' + content }}
#         {%- endif %}
#     {%- endif %}
# {%- endfor %}
# {%- if add_generation_prompt %}
#     {{- '<|im_start|>assistant\n' }}
#     {%- if enable_thinking is defined and enable_thinking is false %}
#         {{- '<think>\n\n</think>\n\n' }}
#     {%- endif %}
# {%- endif %}
# """.strip()

# ��ȡ���������ݼ�
with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

import concurrent.futures
import multiprocessing
import atexit
from math_verify import parse, verify
from math_verify.errors import TimeoutException
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

# ȫ�ֽ��̳أ����Գ�ʼ����
_executor = None

def get_executor():
    global _executor
    if _executor is None and multiprocessing.current_process().name == 'MainProcess':
        _executor = concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
    return _executor

def shutdown_executor():
    global _executor
    if _executor is not None:
        _executor.shutdown()
        _executor = None

atexit.register(shutdown_executor)

def _compute_score_impl(solution_str: str, ground_truth: str) -> float:
    """ʵ�ʼ�������ĺ��ĺ��������ӽ�����ִ�У�"""
    gold_extraction_config = (LatexExtractionConfig(),)
    pred_extraction_config = (LatexExtractionConfig(boxed_match_priority=0, normalization_config=NormalizationConfig(basic_latex=True, units=True, malformed_operators=False, nits=False, boxed="all", equations=False)),)
    
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
        pass  # ���Գ����쳣
    except TimeoutException:
        pass  # ��ʱ����0.0
    return ret_score

def compute_score(solution_str: str, ground_truth: str) -> float:
    """�̰߳�ȫ�ķ����������"""
    executor = get_executor()
    if executor is None:
        # �����ӽ��̣�ֱ�Ӽ���
        return _compute_score_impl(solution_str, ground_truth)
    else:
        # �����̣��ύ�����̳�
        future = executor.submit(_compute_score_impl, solution_str, ground_truth)
        return future.result()

# ��ʼ���������ļ�д��ͽ���������
file_lock = Lock()
pbar = tqdm(total=len(dataset), desc="Processing samples")
pbar_lock = Lock()

def process_sample(instance):

    origin = instance["question"]
    modified = instance["modified"]
    ref = instance["answer"].split("####")[1].strip()
    corr_cnt = 0
    rollouts = []
    for i in range(16):
        rollout = {}
        rollouts.append(rollout)
        try:
            messages = [{"role": "user", "content": init_prompt.replace("[Question]", modified)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
            if DEBUG:
                print("## Example:", messages)
            response = client.completions.create(
                model=MODEL,
                prompt=prompt,
                temperature=0.8,
                max_tokens=2048,
            )
            response = response.choices[0].text
            rollout["r1_reply"] = response
            if DEBUG:
                print("## Response:", (response,))
            hyp = extract_boxed(response)
            question = None
            if hyp:
                hyp = extract_text(hyp)
                if hyp != "None" and len(hyp.split()) >= 4 and "?" in hyp:
                    question = hyp
                else:
                    corr = compute_score(response, ref)
                    if corr:
                        break
                    
            if question is None:
                continue

            messages = [{"role": "user", "content": agent_prompt.replace("[Context]", origin).replace("[Question]", question)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if DEBUG:
                print("## Example:", messages)
            response = client.completions.create(
                model=MODEL,
                prompt=prompt,
                temperature=0.6,
                max_tokens=64,
            )
            response = response.choices[0].text
            rollout["user_reply"] = response
            if DEBUG:
                print("## Response:", (response,))
            
            if response == "None":
                continue
            
            messages = [{"role": "user", "content": init_prompt.replace("[Question]", modified)}, {"role": "assistant", "content": rollout["r1_reply"]}, {"role": "user", "content": final_prompt.replace("[Answer]", rollout["user_reply"])}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if DEBUG:
                print("## Example:", messages)
                print("## Prompt:", (prompt,))
            response = client.completions.create(
                model=MODEL,
                prompt=prompt,
                temperature=0.6,
                max_tokens=4096,
            )
            response = response.choices[0].text
            rollout["r2_reply"] = response
            if DEBUG:
                print("## Response:", (response,))
            corr = compute_score(response, ref)
            rollout["correct"] = corr
            if DEBUG:
                print("## Correct:", corr)
            if corr:
                corr_cnt += 1
            if corr_cnt > 2:
                break
        except Exception as e:
            print(f"API request failed: {e}")
    instance["rollouts"] = rollouts

    # ���洦����
    with file_lock:
        with jsonlines.open(OUTPUT_PATH, mode="a") as writer:
            writer.write(instance)
    
    # ���½�����
    with pbar_lock:
        pbar.update(1)
    
    return instance

# ʹ���̳߳ز�������
start_time = time.time()

with ThreadPoolExecutor(max_workers=64) as executor:
    # �ύ��������
    futures = [executor.submit(process_sample, sample) for sample in dataset]
    
    # �ȴ�����������ɣ������ʵʱ���棩
    for future in futures:
        future.result()

pbar.close()
total_time = time.time() - start_time
print(f"\nAll samples processed in {total_time:.2f} seconds")
print(f"Results saved to: {OUTPUT_PATH}")