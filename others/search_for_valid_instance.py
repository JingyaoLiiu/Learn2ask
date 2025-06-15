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

DATA_PATH = "../dataset/full_train_data.jsonl"
OUTPUT_PATH = "../dataset/full_train_data_search.jsonl"
MODEL = "/public/home/ldk/model_cards/Qwen3-8B"
DEBUG = False

# 初始化API客户端
client = OpenAI(api_key="universe", base_url="http://localhost:8000/v1")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# 读取并采样数据集
with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

import concurrent.futures
import multiprocessing
import atexit
from math_verify import parse, verify
from math_verify.errors import TimeoutException
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

# 全局进程池（惰性初始化）
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
    """实际计算分数的核心函数（在子进程中执行）"""
    gold_extraction_config = (LatexExtractionConfig(),)
    pred_extraction_config = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())
    
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

    origin = instance["origin"]
    modified = instance["modified"]
    ref = instance["answer"].split("####")[1].strip()

    rollouts = []
    for i in range(16):
        rollout = {}
        rollouts.append(rollout)
        try:
            messages = [{"role": "user", "content": init_prompt.replace("[Question]", modified)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if DEBUG:
                print("## Example:", messages)
            response = client.completions.create(
                model=MODEL,
                prompt=prompt,
                temperature=0.8,
                max_tokens=4096,
            )
            response = response.choices[0].text
            rollout["r1_reply"] = response
            if DEBUG:
                print("## Response:", (response,))
            hyp = extract_boxed(response)
            question = None
            if hyp:
                hyp = extract_text(hyp)
                if hyp != "None" and len(hyp.split()) >= 4 and ("?" in hyp[-5:] or hyp.startswith("We need") or hyp.startswith("Please provide") or hyp.startswith("How") or hyp.startswith("What") or hyp.startswith("Please specify")):
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
            messages = [{"role": "user", "content": init_prompt.replace("[Question]", modified)}, {"role": "assistant", "content": rollout["r1_reply"]}, {"role": "user", "content": final_prompt.replace("[Answer]", rollout["user_reply"])}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if DEBUG:
                print("## Example:", messages)
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
            if corr and i > 3:
                break
        except Exception as e:
            print(f"API request failed: {e}")
    instance["rollouts"] = rollouts

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