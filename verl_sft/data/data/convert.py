# -*- coding: utf-8 -*-
import pandas as pd

# ��ȡ parquet �ļ�
df = pd.read_parquet("/public/home/ldk/users/ljy/learn2ask/verl_sft/data/rl_train_raw_rollout16_Meta-Llama-3.1-8B-Instruct.parquet")

# ��ȡǰ 4 ��
#head_df = df.head(4)

# ��ӡ���ն�
print(head_df)

# ���浽�ļ�
head_df.to_csv("/public/home/ldk/users/ljy/learn2ask/verl_sft/data/rl_train_raw_rollout16_Meta-Llama-3.1-8B-Instruct.csv", index=False, encoding="utf-8")
