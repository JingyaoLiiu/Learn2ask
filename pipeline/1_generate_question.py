import sys
import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# MODEL = "/public/home/ldk/model_cards/Qwen3-1.7B"
# DATA_PATH = "/public/home/ldk/users/wat/learn2ask/dataset/full_test_data_filtered.jsonl"
# OUTPUT_PATH = "/public/home/ldk/users/wat/learn2ask/dataset/test_clean_data_step1.jsonl"
MODEL = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
PROMPT_TYPE = "critic"
ENABLE_THINKING = False

prompt_dict = {
    "base": "Please reason step by step, and put your final answer within \\boxed{}.", 
    "critic": "If the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question within \\boxed{}.",
    "final": "If the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question inside \\boxed{}. If no further question can be asked, respond with \\boxed{None}."
}

def extract_text(content):
    """提取 \\text{} 中的内容"""
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

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)[0:1]

queries = []
for instance in dataset:
    # for modified question
    query = "Question: " + instance["modified"] + "\n\n" + prompt_dict[PROMPT_TYPE]
    # for original question
    # query = "Question: " + instance["origin"] + "\n\n" + prompt_dict[PROMPT_TYPE]
    query = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": query},], tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
    queries.append(query)

print((queries[0],))

llm = LLM(
            model=MODEL,
            swap_space=8,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=4096 + 1024,
            gpu_memory_utilization=0.8,
            trust_remote_code=True
        )

sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0,
            max_tokens=4096,
            n=1,
            skip_special_tokens=False,
            include_stop_str_in_output=False
        )
results = llm.generate(queries, sampling_params, use_tqdm=True)
# print(results[0])
for instance, result in zip(dataset, results):
    questions = set()
    for output in result.outputs:
        hyp = extract_boxed(output.text)
        if hyp:
            hyp = extract_text(hyp)
            # whether is a question
            if len(hyp.split()) >= 4 and ("?" in hyp[-5:] or hyp.startswith("We need") or hyp.startswith("Please provide") or hyp.startswith("How") or hyp.startswith("What") or hyp.startswith("Please specify")):
                questions.add(hyp)
    instance["questions"] = list(questions)
    instance["generated_text"] = [output.text for output in result.outputs]

print("Save", len(dataset))

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(dataset)
