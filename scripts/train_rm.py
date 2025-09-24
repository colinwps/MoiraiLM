# path: scripts/train_rm.py
"""
训练奖励模型（Reward Model）
用法：
    python scripts/train_rm.py workdir/spm_shuihu.model workdir/gpt_sft.pth data/rm_data.jsonl workdir/gpt_rm.pth
"""

import sys
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm
from train_gpt import GPTLike, DEVICE, BLOCK_SIZE  # 导入之前的模型定义

# RLHF参数
RM_EPOCHS = 1
RM_LR = 1e-6  # 奖励模型训练通常使用更小的学习率


class RewardModel(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        # 使用GPTLike的结构，但移除最后一个线性层
        self.transformer = GPTLike(vocab_size, block_size)
        # 添加一个输出奖励值的线性层
        self.head = nn.Linear(self.transformer.head.in_features, 1)
        # 移除GPTLike的原始head
        del self.transformer.head

    def forward(self, idx):
        # 拿到Transformer的输出
        x = self.transformer(idx)
        # 只取最后一个token的输出作为奖励
        return self.head(x[:, -1, :])


class RMDataset(Dataset):
    def __init__(self, token_ids):
        self.ids = token_ids

    def __len__(self):
        # RM数据是问答对的对比排序，所以我们将多个对比样本拼接
        return len(self.ids)

    def __getitem__(self, idx):
        # 每个样本包含一个提示和两个不同质量的回答
        # data format: {"prompt":..., "chosen":..., "rejected":...}
        # token_ids: [prompt_ids, chosen_ids], [prompt_ids, rejected_ids]
        prompt_ids = self.ids[idx]["prompt_ids"]
        chosen_ids = self.ids[idx]["chosen_ids"]
        rejected_ids = self.ids[idx]["rejected_ids"]

        # 将prompt和回答拼接
        chosen_input = torch.tensor(prompt_ids + chosen_ids, dtype=torch.long)
        rejected_input = torch.tensor(prompt_ids + rejected_ids, dtype=torch.long)
        return chosen_input, rejected_input


def create_rm_dataset(sp, data_path):
    all_pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prompt = data["prompt"]
            chosen = data["chosen"]
            rejected = data["rejected"]

            # 将每个部分编码成 token IDs
            prompt_ids = sp.encode(prompt, out_type=int)
            chosen_ids = sp.encode(chosen, out_type=int) + [sp.eos_id()]
            rejected_ids = sp.encode(rejected, out_type=int) + [sp.eos_id()]

            all_pairs.append({
                "prompt_ids": prompt_ids,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids
            })
    return RMDataset(all_pairs)


def train_rm(model, dataset, epochs=RM_EPOCHS, lr=RM_LR):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    pbar = tqdm(loader, desc="RM Training")
    for epoch in range(epochs):
        for chosen_input, rejected_input in pbar:
            # 确保输入长度不超过BLOCK_SIZE
            chosen_input = chosen_input[:, -BLOCK_SIZE:].to(DEVICE)
            rejected_input = rejected_input[:, -BLOCK_SIZE:].to(DEVICE)

            chosen_reward = model(chosen_input)
            rejected_reward = model(rejected_input)

            # 使用二分类交叉熵损失，目标是让chosen的奖励高于rejected
            # 这里简化为直接比较两个奖励值，并使用损失函数进行优化
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             chosen_r=f"{chosen_reward.item():.2f}",
                             rejected_r=f"{rejected_reward.item():.2f}")


def main():
    if len(sys.argv) < 5:
        print("用法: python scripts/train_rm.py 分词器模型 SFT模型 RM数据 输出模型")
        sys.exit(1)

    sp_model_path, sft_model_path, rm_data_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    vocab_size = sp.get_piece_size()

    # 1. 加载SFT模型并转换为奖励模型
    sft_model = GPTLike(vocab_size, BLOCK_SIZE).to(DEVICE)
    sft_model.load_state_dict(torch.load(sft_model_path, map_location=DEVICE))

    rm_model = RewardModel(vocab_size, BLOCK_SIZE).to(DEVICE)
    # 将SFT模型的权重加载到RM模型中，除了最后的head
    rm_model.transformer.load_state_dict(sft_model.state_dict(), strict=False)
    print(f"成功加载SFT模型并初始化奖励模型: {sft_model_path}")

    # 2. 准备奖励模型数据集
    rm_dataset = create_rm_dataset(sp, rm_data_path)
    print(f"RM数据集已准备，共 {len(rm_dataset)} 个排序对")

    # 3. 训练奖励模型
    print("开始训练奖励模型...")
    train_rm(rm_model, rm_dataset)

    torch.save(rm_model.state_dict(), out_path)
    print(f"奖励模型已保存: {out_path}")


if __name__ == "__main__":
    main()