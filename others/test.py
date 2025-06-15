import jsonlines
from transformers import AutoTokenizer
import re

data_path = "test.jsonl"
model_path = "/public/home/ldk/model_cards/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

with jsonlines.open(data_path) as reader:
    dataset = list(reader)

text = """<|im_start|>user\nQuestion: "Alyssa, Keely, and Kendall ordered 100 chicken nuggets from a fast-food restaurant. How many did Alyssa eat?"\n\nIf the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question inside \\boxed{}. If no further question can be asked, respond with \\boxed{None}.<|im_end|>\n<|im_start|>assistant\n"""

def extract_question(text):
    # 定义正则表达式模式：匹配 Question: 和 If the question is 之间的内容
    pattern = r'Question:\s*(.*?)\s*If the question is answerable,'
    
    # 使用re.DOTALL标志确保.匹配换行符，re.IGNORECASE忽略大小写差异
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # 返回捕获组内容并去除首尾空白
        return match.group(1).strip()
    else:
        return None

print(extract_question(text))
exit()


from tqdm import tqdm

data_path = "full_test_data_critic_false.jsonl"
with jsonlines.open(data_path, "r") as reader:
    dataset = list(reader)

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

try:
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from latex2sympy2_extended.latex2sympy2 import NormalizationConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


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

doubt = 0
invalid = 0
corr, total = 0, 0
doubt_pass_k, corr_pass_k = 0, 0
doubt_cnt_dict, corr_cnt_dict = {}, {}

for instance in tqdm(dataset):
    ref = instance["answer"].split("####")[1].strip()
    doubt_cnt, corr_cnt = 0, 0
    for generated_text in instance["generated_text"]:
        hyp = extract_boxed(generated_text)
        if hyp:
            if len(hyp.split()) >= 4 and ("?" in hyp[-5:] or hyp.startswith("We need") or hyp.startswith("\\text{We need") or hyp.startswith("Please provide") or hyp.startswith("\\text{Please provide") or hyp.startswith("How") or hyp.startswith("\\text{How")) or hyp.startswith("What") or hyp.startswith("\\text{What") or hyp.startswith("Please specify") or hyp.startswith("\\text{Please specify"):
                doubt += 1
                doubt_cnt += 1
            else:
                if compute_score(hyp, ref):
                    corr += 1
                    corr_cnt += 1
        total += 1
    if doubt_cnt > 0:
        doubt_pass_k += 1
    if corr_cnt > 0:
        corr_pass_k += 1
    
    if doubt_cnt in doubt_cnt_dict:
        doubt_cnt_dict[doubt_cnt] += 1
    else:
        doubt_cnt_dict[doubt_cnt] = 1
    if corr_cnt in corr_cnt_dict:
        corr_cnt_dict[corr_cnt] += 1
    else:
        corr_cnt_dict[corr_cnt] = 1

print(doubt/total ,doubt, total)
print(corr/total, corr, total - doubt - invalid)

print(invalid, total)

print(doubt_pass_k / len(dataset), doubt_pass_k, len(dataset))
print(corr_pass_k / len(dataset), corr_pass_k, len(dataset))

print("doubt")
for key in sorted(doubt_cnt_dict.keys()):
    print(key, doubt_cnt_dict[key] / sum(doubt_cnt_dict.values()))

print("corr")
for key in sorted(corr_cnt_dict.keys()):
    print(key, corr_cnt_dict[key] / sum(corr_cnt_dict.values()))
