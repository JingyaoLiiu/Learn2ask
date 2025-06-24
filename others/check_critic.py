import jsonlines
from tqdm import tqdm

data_path = "full_test_data_critic_true.jsonl"
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
            if (len(hyp.split()) >= 4 or "?" in hyp[-5:]):
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
