import sys
import jsonlines
from transformers import AutoTokenizer

MODEL = sys.argv[1]
DATA_PATH = sys.argv[2]

PROMPT_TYPE = sys.argv[3]
ENABLE_THINKING = True if sys.argv[4] == "true" else False

OUTPUT_PATH = sys.argv[5]

prompt_dict = {
    "base": "Please reason step by step, and put your final answer within \\boxed{}.",
    "critic": "If the question is answerable, provide the final answer within \\boxed{}. Otherwise, ask the user for the necessary information by phrasing the request as a question within \\boxed{}."
}

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

for instance in dataset:
    query = "Question: " + instance["modified"] + "\n\n" + prompt_dict[PROMPT_TYPE]
    instance["query"] = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": query},], tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)

print(instance["query"])
print("Save", len(dataset))

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(dataset)
