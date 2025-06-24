
import torch
from transformers import AutoTokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

if __name__ == "__main__":
    # Step 1: 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/public/home/ldk/model_cards/Llama-3.2-3B-Instruct")

    # Step 2: 加载你的 parquet 文件路径
    parquet_file = "/public/home/ldk/users/ljy/l2a/sft/data/rl_train_raw_rollout16_Meta-Llama-3.1-8B-Instruct.parquet"

    # Step 3: 配置 config
    config = {
        "truncation": "error",
        "max_length": 1024,
        "multiturn": {
            "messages_key": "messages",
            "tools_key": "tools",  # 如果没有工具你可以删掉这行
            "enable_thinking_key": "enable_thinking",  # 如果你新增了这一列
        },
    }

    # Step 4: 初始化数据集
    dataset = MultiTurnSFTDataset(parquet_file, tokenizer, config=config)

    # Step 5: 查看数据集长度
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Step 6: 取第一个样本并查看输出结构
    sample = dataset[0]
    print("sample[0]:",sample)
    print("Sample keys:", sample.keys())
    print("input_ids shape:", sample["input_ids"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)
    print("position_ids shape:", sample["position_ids"].shape)
    print("loss_mask shape:", sample["loss_mask"].shape)

    # 可视化 decode 出来看看输入
    decoded = tokenizer.decode(sample["input_ids"][sample["attention_mask"] == 1])
    print("Decoded input:\n", repr(decoded))

    decoded = tokenizer.decode(sample["input_ids"][sample["loss_mask"] == 1])
    print("Decoded valid loss input:\n", repr(decoded))


