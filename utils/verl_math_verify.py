# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from latex2sympy2_extended.latex2sympy2 import NormalizationConfig
    from math_verify import (
        parse,
        verify,
        ExprExtractionConfig,
        LatexExtractionConfig,
        StringExtractionConfig,
    )
except ImportError:
    print(
        "To use Math-Verify, please install it first by running `pip install math-verify`."
    )
import concurrent.futures

TIMEOUT_SECONDS = 30


def parse_label(label):
    return parse(
        label,
        [
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ],
    )


def parse_prediction(prediction):
    return parse(
        prediction,
        [
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
            StringExtractionConfig(),
        ],
    )


def verify_with_timeout(verify_func, gold_latex, pred_latex, timeout=TIMEOUT_SECONDS):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(verify_func, gold_latex, pred_latex)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("Verification timed out!")
            return 0.0, None  


def extract_boxed(content):
    """提取 \\boxed{} 中的内容"""
    start_index = content.rfind("\\boxed{")
    if start_index == -1:
        return None

    start_index += len("\\boxed{")
    brace_count = 1
    end_index = start_index

    while end_index < len(content) and brace_count > 0:
        if content[end_index] == "{":
            brace_count += 1
        elif content[end_index] == "}":
            brace_count -= 1
        end_index += 1

    if brace_count != 0:
        return None

    return content[start_index : end_index - 1].strip()


def split_solution_str(solution_str):
    assistant1_part = solution_str
    user_part = ""
    assistant2_part = ""
    user_split = "user\nAnswer to your question"
    assistant2_split = "assistant\n<think>\n\n</think>\n\n"
    user_begin = solution_str.find(user_split)
    assistant2_begin = solution_str.find(assistant2_split)
    if user_begin != -1:
        assistant1_part = solution_str[:user_begin]
        if assistant2_begin != -1:
            user_part = solution_str[user_begin:assistant2_begin]
            assistant2_part = solution_str[assistant2_begin:]
        else:
            user_part = solution_str[user_begin:]
    return assistant1_part, user_part, assistant2_part


FORMAT_REWARD = 0.5
ACC_REWARD_FIRST_ROUND = 1.0
ACC_REWARD_SECOND_ROUND = 0.8

def compute_score(
    solution_str: str, ground_truth: str, data_source=None, extra_info=None
) -> float:
    # support normal and missing-condition gsm8k
    # split solution_str into assistant1, user and assistant2 parts
    assistant1_part, user_part, assistant2_part = split_solution_str(solution_str)
    score = 0.0
    if assistant2_part:
        maybe_q = extract_boxed(assistant1_part)
        maybe_ans = extract_boxed(assistant2_part)
        if maybe_q and maybe_ans:
            score += FORMAT_REWARD
        try:
            ground_truth_boxed = "\\boxed{" + ground_truth + "}"
            parsed_label = parse_label(ground_truth_boxed)
            parsed_prediction = parse_prediction(assistant2_part)
            verify_res = verify(parsed_label, parsed_prediction)
            if not verify_res:
                ground_truth_boxed = "\\boxed{" + ground_truth + "%}"
                parsed_label = parse_label(ground_truth_boxed)
                verify_res = verify(parsed_label, parsed_prediction)
            score += ACC_REWARD_SECOND_ROUND  if verify_res else 0.0
        except Exception:
            pass
    else:
        first_ans = extract_boxed(assistant1_part)
        if first_ans is not None and first_ans:
            score += FORMAT_REWARD
        try:
            ground_truth_boxed = "\\boxed{" + ground_truth + "}"
            parsed_label = parse_label(ground_truth_boxed)
            parsed_prediction = parse_prediction(assistant1_part)
            verify_res = verify(parsed_label, parsed_prediction)
            if not verify_res:
                ground_truth_boxed = "\\boxed{" + ground_truth + "%}"
                parsed_label = parse_label(ground_truth_boxed)
                verify_res = verify(parsed_label, parsed_prediction)
            score += ACC_REWARD_FIRST_ROUND  if verify_res else 0.0
        except Exception:
            pass
    return score


