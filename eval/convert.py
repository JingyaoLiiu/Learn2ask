# import pandas as pd

# # 指定你的 parquet 文件路径
# parquet_path = "/public/home/ldk/model_cards/gsm8k/main/test-00000-of-00001.parquet"

# # 读取 parquet 文件为 DataFrame
# df = pd.read_parquet(parquet_path)

# # 保存为 JSON 文件（行格式）
# json_path = "/public/home/ldk/users/ljy/l2a/eval/data/origin_gsm_test.json"
# df.to_json(json_path, orient="records", lines=True, force_ascii=False)

# print(f"成功将 {parquet_path} 转换为 {json_path}")

import json

input_path = "/public/home/ldk/users/ljy/l2a/eval/origin_gsm_test.json"
output_path = "/public/home/ldk/users/ljy/l2a/eval/data/origin_gsm_test.json"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        data["modified"] = data["question"]  # 新增字段 modified，复制 question 字段内容
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"已保存新文件到: {output_path}")
