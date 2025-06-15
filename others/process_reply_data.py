import sys
import jsonlines
from transformers import AutoTokenizer
import copy

MODEL = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
ENABLE_THINKING = True if sys.argv[4] == "true" else False
SNOWBALL = True if sys.argv[5] == "true" else False

first_round_prompt = "Question: [QUESTION]\n\nIf the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question within \\boxed{}."

if SNOWBALL:
    second_round_prompt = "Answer to your question: [ANSWER]\n\nWith these information, answer the previous question: [QUESTION]"
else:
    second_round_prompt = "Answer to your question: [ANSWER]\n\nWith these information, answer the previous question."

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

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

output = []
for instance in dataset:
    user_1 = first_round_prompt.replace("[QUESTION]", instance["modified"])
    assistant_1 = extract_text(instance["user_query"])
    if SNOWBALL:
        user_2 = second_round_prompt.replace("[ANSWER]", instance["agent_reply"]).replace("[QUESTION]", instance["modified"])
    else:
        user_2 = second_round_prompt.replace("[ANSWER]", instance["agent_reply"])
    instance["query"] = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": user_1}, {"role": "assistant", "content": assistant_1}, {"role": "user", "content": user_2}], tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
    output.append(instance)

print(output[0]["query"])
print("Save", len(output))

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(output)
