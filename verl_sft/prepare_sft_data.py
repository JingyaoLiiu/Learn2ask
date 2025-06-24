import jsonlines
from search_utils import *
from datasets import load_dataset, Dataset

data_fpath_list = [
    "/public/home/ldk/users/ljy/learn2ask/verl_sft/data/Llama-3.1-8B-Instruct_train_3k_rollout_normal.jsonl",
    "/public/home/ldk/users/ljy/learn2ask/verl_sft/data/Llama-3.1-8B-Instruct_train_3k_rollout_learn2ask.jsonl",
    "/public/home/ldk/users/ljy/learn2ask/verl_sft/data/Llama-3.1-8B-Instruct_train_3k_rollout_hint.jsonl"
]
dataset = []
for data_fpath in data_fpath_list:
    with jsonlines.open(data_fpath) as reader:
        for instance in reader:
            question = instance["modified"] if "modified" in instance else instance["question"]
            r1_prompt = init_prompt if data_fpath.endswith("learn2ask") or data_fpath.endswith("normal") else doubt_prompt
            query = r1_prompt.replace("[Question]", question)
            valid_rollouts = [rollout for rollout in instance["rollouts"] if rollout.get("correct", False)]
            if valid_rollouts:
                # check validation
                for rollout in valid_rollouts:
                    r1_reply = rollout["r1_reply"]
                    if "</think>" in r1_reply:
                        solution = r1_reply.split("</think>")[1]
                        answer = extract_boxed(solution)
                        if answer is None or len(answer) == 0:
                            continue
                        messages = [{"role": "user", "content": query}, {"role": "assistant", "content": r1_reply}]
                        if "r2_reply" in rollout:
                            user_reply = rollout["user_reply"]
                            if "None" in user_reply:
                                continue
                            r2_reply = rollout["r2_reply"]
                            messages.append({"role": "user", "content": final_prompt.replace("[Answer]", user_reply)})
                            messages.append({"role": "assistant", "content": r2_reply})
                        dataset.append({"messages": messages})
                        break
                    else:
                        solution = r1_reply
                        answer = extract_boxed(solution)
                        if answer is None or len(answer) == 0:
                            continue
                        messages = [{"role": "user", "content": query}, {"role": "assistant", "content": r1_reply}]
                        if "r2_reply" in rollout:
                            user_reply = rollout["user_reply"]
                            if "None" in user_reply:
                                continue
                            r2_reply = rollout["r2_reply"]
                            messages.append({"role": "user", "content": final_prompt.replace("[Answer]", user_reply)})
                            messages.append({"role": "assistant", "content": r2_reply})
                        dataset.append({"messages": messages})
                        break

    print(len(dataset))

output = Dataset.from_list(dataset)
output.to_parquet(f"/public/home/ldk/users/ljy/learn2ask/verl_sft/data/llama_train_with_hint.parquet")