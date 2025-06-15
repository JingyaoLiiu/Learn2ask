import jsonlines
import numpy as np
from tqdm import tqdm
from math_verify import parse, verify
import re
import string

data_path = "/public/home/ldk/users/ljy/learn2ask/sft/data/Qwen3-4B_test_step3.jsonl"
with jsonlines.open(data_path, "r") as reader:
    dataset = list(reader)

try:
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from latex2sympy2_extended.latex2sympy2 import NormalizationConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

def remove_text_commands(text):
    # 正则表达式模式：匹配 \text{...} 
    pattern = r'\\text\{.*?\}'  
    # 使用非贪婪匹配，并添加 re.DOTALL 确保跨行生效
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text

def compute_score(solution_str: str, ground_truth: str) -> bool:
    gold_extraction_config = (LatexExtractionConfig(),)
    pred_extraction_config = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())

    ret_score = 0.0
    timeout_score = 0.0
    # Wrap the ground truth in \boxed{} format for verification
    try:
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        gold = parse(ground_truth_boxed, gold_extraction_config)
        pred = parse(solution_str, pred_extraction_config)
        ret_score = verify(gold, pred)
        if not ret_score:
            ground_truth_boxed = "\\boxed{" + ground_truth + "%}"
            gold = parse(ground_truth_boxed, gold_extraction_config)
            ret_score = verify(gold, pred)
    except Exception as e:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score

def extract_text(content):
    if "\\text{" in content:
        pass
    else:
        return content

    start_index = content.rfind('\\text{')
    start_index += len('\\text{')
    if start_index == -1:
        return None  # No opening brace found
    
    brace_count = 1
    end_index = start_index
    
    while end_index < len(content) and brace_count > 0:
        end_index += 1
        if content[end_index] == '{':
            brace_count += 1
        elif content[end_index] == '}':
            brace_count -= 1
    if brace_count != 0:
        return None  # Unbalanced braces
    
    return content[start_index:end_index]

def extract_boxed(content):
    """提取 \\boxed{} 中的内容"""
    start_index = content.rfind('\\boxed{')
    if start_index == -1:
        return None
    
    start_index += len('\\boxed{')
    brace_count = 1
    end_index = start_index
    
    while end_index < len(content) and brace_count > 0:
        if content[end_index] == '{':
            brace_count += 1
        elif content[end_index] == '}':
            brace_count -= 1
        end_index += 1
    
    if brace_count != 0:
        return None
    
    return content[start_index:end_index-1].strip()

def extract_numbers(text):
    pattern = r'[-+]?(?:\d+\.\d+|\d+\.?|\.\d+)'
    numbers = re.findall(pattern, text)
    return numbers

final_corr = 0
request_cnt = 0
dontknow_cnt = 0
total = 0
output = []
for instance in tqdm(dataset):
    ref = instance["answer"].split("####")[1].strip()

    for rollout in instance["rollout"]:
        total += 1
        response = rollout["r1_reply"]
        boxed = extract_boxed(response)
        if boxed:
            boxed = extract_text(boxed)
            if boxed:
                if boxed == "None":
                    dontknow_cnt += 1
                elif "user_reply" in rollout:
                    request_cnt += 1
                    response = rollout["r2_reply"]
        if compute_score(response, ref):
            final_corr += 1

print(final_corr / total, final_corr, total)
print(request_cnt / total, request_cnt, total)
#print(dontknow_cnt / total, dontknow_cnt, total)
