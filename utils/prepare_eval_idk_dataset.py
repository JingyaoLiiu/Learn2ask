import jsonlines
import numpy as np
from tqdm import tqdm
from math_verify import parse, verify
import re
import string
from datasets import load_dataset, Dataset

data_path = "/public/home/ldk/users/wat/learn2ask/dataset/test_idk.jsonl"
with jsonlines.open(data_path, "r") as reader:
    dataset = list(reader)

prompt = "Question: [Question]\n\nIf the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question inside \\boxed{}. If no further question can be asked, respond with \\boxed{None}."

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

# add a row to each data item that represents a unique id
def process_fn(example):
    origin = example.pop('origin')
    marked = example.pop("marked")
    modified = example.pop('modified')

    question = prompt.replace("[Question]", modified)

    answer = example.pop('answer')
    solution = extract_solution(answer)
    data = {
        "data_source": "gsm8k",
        "prompt": [{
            "role": "user",
            "content": question
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            "origin": origin,
            "marked": marked,
            "modified": modified,
            "answer": answer
        }
    }
    return data

output = []
for instance in tqdm(dataset):
    output.append(process_fn(instance))

print(len(output))

with jsonlines.open(f"test.jsonl", "w") as writer:
    writer.write_all(output)

output = Dataset.from_list(output)
output.to_parquet(f"test.parquet")