# path: scripts/sft_gpt.py
"""
对 GPT 模型进行监督微调 (SFT)
用法：
    python scripts/sft_gpt.py workdir/spm_shuihu.model workdir/gpt_shuihu_RoPE.pth data/qa_data.jsonl workdir/gpt_sft.pth
"""

import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_gpt import GPTLike, DEVICE, BLOCK_SIZE, train, clean_text  # 导入预训练脚本中的相关组件
import sentencepiece as spm

# 监督微调参数
SFT_EPOCHS = 3
SFT_LR = 1e-5  # SFT通常使用更小的学习率


class SFTDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.ids = token_ids
        self.block_size = block_size

    def __len__(self):
        # 确保有足够的序列长度
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, idx):
        # input: [idx, ..., idx + block_size - 1]
        # target: [idx + 1, ..., idx + block_size]
        x = torch.tensor(self.ids[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1: idx + 1 + self.block_size], dtype=torch.long)
        return x, y


def create_sft_dataset(sp, data_path):
    """从JSONL文件中加载并处理SFT数据"""
    all_token_ids = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            instruction = data.get("instruction", "")
            output = data.get("output", "")

            # 使用特殊标记来区分指令和回答
            # 常见格式：<|user|>用户问题<|endofuser|><|assistant|>模型回答<|endofassistant|>
            # 这里简化为：问题 + 分隔符 + 回答
            prompt = clean_text(f"{instruction}。{output}")  # 问号/句号作为分隔符

            # 编码成token ID
            token_ids = sp.encode(prompt, out_type=int)
            all_token_ids.extend(token_ids)
            # 添加一个特殊的结束符，以区分不同的问答对
            all_token_ids.append(sp.eos_id())

    return SFTDataset(all_token_ids, BLOCK_SIZE)


def main():
    if len(sys.argv) < 5:
        print("用法: python scripts/sft_gpt.py 分词器模型 预训练模型 SFT数据 输出模型")
        sys.exit(1)

    sp_model_path, pretrained_model_path, sft_data_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    # 1. 加载分词器和预训练模型
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    vocab_size = sp.get_piece_size()
    model = GPTLike(vocab_size, BLOCK_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=DEVICE))
    print(f"成功加载预训练模型: {pretrained_model_path}")

    # 2. 准备SFT数据集
    sft_dataset = create_sft_dataset(sp, sft_data_path)
    print(f"SFT数据集已准备，共 {len(sft_dataset)} 条数据")

    # 3. 监督微调
    print("开始监督微调...")
    train(model, sft_dataset, epochs=SFT_EPOCHS, lr=SFT_LR)

    # 4. 保存微调后的模型
    torch.save(model.state_dict(), out_path)
    print(f"微调后的模型已保存: {out_path}")


if __name__ == "__main__":
    main()