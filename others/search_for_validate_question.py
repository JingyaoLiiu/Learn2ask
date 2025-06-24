import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/public/home/ldk/model_cards/Qwen3-8B"
DATA_PATH = "full_train_data.jsonl"
OUTPUT_PATH = "full_train_data_search_FF.jsonl"
Q_ENABLE_THINKING = False
A_ENABLE_THINKING = False

init_prompt = "Question: [QUESTION]\n\nIf the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question within \\boxed{}."

agent_prompt = """
## Role
You are a **Math Question Analyzer**, a specialized AI assistant designed to extract and provide specific information from given math problems based on student queries.

## Capabilities
- Analyze the content of the provided math question with precision  
- Identify and extract requested information if relevant parts present in the question  

## Knowledge Base
- Mathematical terminology and problem structures  
- Information extraction techniques  
- Contextual understanding of student inquiries  

## Instructions
1. **Input Format**:  
   - Math question: "[math_question]"
   - Student's query: "[student_question]"

2. **Processing Rules**:  
   - Output **only the combined relevant parts about the requested information** (no explanations).  
   - Output "None" If the requested information is not mentioned in the math problem.

3. **Constraints**:  
   - Never infer or calculate missing information.  
   - Never add commentary, examples, or supplemental text.  
   - Prioritize brevity and accuracy.  

## Output Example
**Math question**: "James earns $20 an hour while working at his main job. He earns 20% less while working his second job. He works 30 hours at his main job and half that much at his second job. How much does he earn per week?"
**Student's query**: "What is James's hourly wage at his second job?"
**Reply**: "20% less than his main job"

## Your Turn
**Math question**: "[Context]"
**Student's query**: "[Question]"
**Reply**:
""".strip()

final_prompt = "Answer to your question: [ANSWER]\n\nWith these information, answer my previous question."

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


tokenizer = AutoTokenizer.from_pretrained(MODEL)

with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

llm = LLM(
        model=MODEL,
        swap_space=8,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=4096 + 1024,
        gpu_memory_utilization=0.8,
        trust_remote_code=True
    )

for _round in range(4):
    print("## Round: ", _round)
    # 1st round
    anchors = []
    queries = []
    for instance in dataset:
        corr = 0
        for r2_output in instance.get("r2_output", []):
            for rollout in r2_output.get("rollout", []):
                if rollout["corr"]:
                    corr += 1
        if corr > 2:
            continue
        query = init_prompt.replace("[QUESTION]", instance["modified"])
        query = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": query},], tokenize=False, add_generation_prompt=True, enable_thinking=Q_ENABLE_THINKING)
        queries.append(query)
        anchors.append(instance)

    print("## Example:", (queries[0],))

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        min_p=0,
        max_tokens=4096,
        n=8,
        skip_special_tokens=False,
        include_stop_str_in_output=True
    )
    results = llm.generate(queries, sampling_params, use_tqdm=True)
    
    r2_anchors = []
    r2_queries = []
    for anchor, result in zip(anchors, results):
        if "r1_output" not in anchor:
            anchor["r1_output"] = []
        if "r1_result" not in anchor:
            anchor["r1_question"] = []
        if "r2_output" not in anchor:
            anchor["r2_output"] = []
        for output in result.outputs:
            anchor["r1_output"].append(output.text)
            hyp = extract_boxed(output.text)
            if hyp:
                hyp = extract_text(hyp)
                # whether is a question
                if len(hyp.split()) >= 4 and ("?" in hyp[-5:] or hyp.startswith("We need") or hyp.startswith("Please provide") or hyp.startswith("How") or hyp.startswith("What") or hyp.startswith("Please specify")):
                    if hyp not in anchor["r1_question"]:
                        anchor["r1_question"].append(hyp)
                        r2_anchor = {"question": hyp}
                        r2_anchors.append(r2_anchor)
                        anchor["r2_output"].append(r2_anchor)
                        r2_queries.append({
                            "origin": anchor["origin"],
                            "modified": anchor["modified"],
                            "user_query": hyp
                        })
    
    agent_queries = []
    for r2_query in r2_queries:
        question = agent_prompt.replace("[Context]", r2_query["origin"].replace("[", "").replace("]", "")).replace("[Question]", r2_query["user_query"])
        agent_queries.append(tokenizer.apply_chat_template(conversation=[{"role": "user", "content": question},], tokenize=False, add_generation_prompt=True, enable_thinking=False))
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        max_tokens=4096,
        n=8,
        skip_special_tokens=False,
        include_stop_str_in_output=True
    )
    print("## Example:", (agent_queries[0],))
    results = llm.generate(agent_queries, sampling_params, use_tqdm=True)
    
    r4_queries, r4_anchors = [], []
    for anchor, r2_query, result in zip(r2_anchors, r2_queries, results):
        if "agent_reply" not in r2_query:
            r2_query["agent_reply"] = []
        for output in result.outputs:
            output = output.text
            if "<|im_end|>" in output:
                output = output.replace("<|im_end|>", "")
            output = output.strip()
            if output and output.lower() != "none":
                user_1 = init_prompt.replace("[QUESTION]", r2_query["modified"])
                assistant_1 = extract_text(r2_query["user_query"])
                user_2 = final_prompt.replace("[ANSWER]", output)
                r4_query = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": user_1}, {"role": "assistant", "content": assistant_1}, {"role": "user", "content": user_2}], tokenize=False, add_generation_prompt=True, enable_thinking=A_ENABLE_THINKING)
                r4_queries.append(r4_query)
                rollout_anchor = {"agent_reply": output,}
                r4_anchors.append(rollout_anchor)
                if "rollout" not in anchor:
                    anchor["rollout"] = []
                anchor["rollout"].append(rollout_anchor)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        max_tokens=4096,
        n=1,
        skip_special_tokens=False,
        include_stop_str_in_output=True
    )
    if len(r4_queries) == 0:
        continue
    print("## Example:", (r4_queries[0],))
    results = llm.generate(r4_queries, sampling_params, use_tqdm=True)
    
    for anchor, result in zip(r4_anchors, results):
        anchor["solution"] = result.outputs[0].text

    # assert correctness
    for instance in dataset:
        ref = instance["answer"].split("####")[1].strip()
        for r2_output in instance["r2_output"]:
            for rollout in r2_output.get("rollout", []):
                hyp = rollout["solution"]
                rollout["corr"] = compute_score(hyp, ref)

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(dataset)
