import jsonlines
import numpy as np
from tqdm import tqdm
from math_verify import parse, verify
import re
import string

data_path = "/public/home/ldk/users/wat/learn2ask/dataset/Qwen3-14B_test_think_step3.jsonl"
# OUTPUT_PATH = "/public/home/ldk/users/wat/learn2ask/dataset/test_clean_data_step4.jsonl"
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

guess = 0
final_corr = 0
total = 0
pass_list = []
output = []
for instance in tqdm(dataset):
    total += 1
    ref = instance["answer"].split("####")[1].strip()
    
    predictions = []
    for response in instance["generated_text"]:
        boxed = extract_boxed(response)
        if boxed:
            if len(boxed.split()) > 3 or "information".lower() in boxed:
                continue
            else:
                num = extract_numbers(boxed.replace(",", ""))
                if num:
                    predictions.append(num[0])
    
    if len(predictions) > 14:
        if len(set(predictions)) < 2:
            guess += 1
            continue

    corr_num, q_num = 0, 0
    for rollout in instance["rollouts"]:
        corr = False
        for solution in rollout["solution"]:
            if compute_score(solution, ref):
                corr = True
                break
        if corr:
            corr_num += 1
        q_num += 1

    if q_num:
        pass_list.append(corr_num / q_num)
    
        if corr_num / q_num < 1:
            output.append({
                "origin": instance["origin"],
                "modified": instance["modified"],
                "marked": instance["marked"],
                "answer": instance["answer"]
            })

    if corr_num:
        final_corr += 1



print(guess / total, guess, total)
print(final_corr / total, final_corr, total)
print(len(pass_list) / total, len(pass_list), total)

pass_array = np.array(pass_list)

print("100%", np.mean(pass_array > 0.99))
print("75%", np.mean(pass_array > 0.74))
print("50%", np.mean(pass_array > 0.49))
print("25%", np.mean(pass_array > 0.24))
print("0%", np.mean(pass_array > 0.))

# diverse
print(len(set(item["origin"] for item in output)))
print(len(output))

# with jsonlines.open(OUTPUT_PATH, "w") as writer:
#     writer.write_all(output)