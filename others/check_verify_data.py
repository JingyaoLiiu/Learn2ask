import jsonlines
import numpy as np

data_path = "full_test_data_search_FF.jsonl"
with jsonlines.open(data_path, "r") as reader:
    dataset = list(reader)

pass_k, total = 0, 0
pass_list = []
for instance in dataset:
    corr = 0
    q_total = 0
    q_pass_list = []
    for r2_output in instance.get("r2_output", []):
        q_corr = 0
        for rollout in r2_output.get("rollout", []):
            if rollout["corr"]:
                corr += 1
                q_corr += 1
        q_total += 1
        q_pass_list.append(q_corr > 0)
    if corr > 0:
        pass_k += 1
    total += 1

    if q_pass_list:
        pass_list.append(np.mean(q_pass_list))
    else:
        pass_list.append(0)

print(pass_k / total, pass_k, total)

pass_array = np.array(pass_list)

print("100%", np.mean(pass_array > 0.99))
print("75%", np.mean(pass_array > 0.74))
print("50%", np.mean(pass_array > 0.49))
print("25%", np.mean(pass_array > 0.24))
print("0%", np.mean(pass_array > 0.))