# -*- coding: utf-8 -*-
import pandas as pd

# 读取 parquet 文件
df = pd.read_parquet("/public/home/ldk/users/ljy/learn2ask/verl_sft/data/rl_train_raw_rollout16_Meta-Llama-3.1-8B-Instruct.parquet")

# 获取前 4 行
#head_df = df.head(4)

# 打印到终端
print(head_df)

# 保存到文件
head_df.to_csv("/public/home/ldk/users/ljy/learn2ask/verl_sft/data/rl_train_raw_rollout16_Meta-Llama-3.1-8B-Instruct.csv", index=False, encoding="utf-8")
