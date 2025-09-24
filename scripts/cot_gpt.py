# path: scripts/cot_gpt.py
"""
对 GPT 模型进行链式思考（CoT）微调
用法：
    python scripts/cot_gpt.py workdir/spm_shuihu.model workdir/gpt_sft.pth data/cot_data.jsonl workdir/gpt_cot.pth
"""

import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm
# 确保导入的是正确的模型定义，比如带RoPE的版本
from train_gpt import GPTLike, DEVICE, BLOCK_SIZE, train, clean_text

# 链式思考微调参数
COT_EPOCHS = 1
COT_LR = 1e-6 # 使用更小的学习率，避免覆盖SFT学到的知识

class CoTDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1: idx + 1 + self.block_size], dtype=torch.long)
        return x, y

def create_cot_dataset(sp, data_path):
    """从JSONL文件中加载并处理CoT数据"""
    all_token_ids = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            instruction = data.get("instruction", "")
            output = data.get("output", "")

            # 拼接成“问题 + 回答”格式
            prompt = clean_text(f"{instruction}{output}")

            # 编码成token ID
            token_ids = sp.encode(prompt, out_type=int)
            all_token_ids.extend(token_ids)
            # 添加特殊的结束符，以区分不同的样本
            all_token_ids.append(sp.eos_id())

    return CoTDataset(all_token_ids, BLOCK_SIZE)

def main():
    if len(sys.argv) < 5:
        print("用法: python scripts/cot_gpt.py 分词器模型 SFT模型 CoT数据 输出模型")
        sys.exit(1)

    sp_model_path, sft_model_path, cot_data_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    # 1. 加载分词器和 SFT 模型
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    vocab_size = sp.get_piece_size()
    model = GPTLike(vocab_size, BLOCK_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(sft_model_path, map_location=DEVICE))
    print(f"成功加载 SFT 模型: {sft_model_path}")

    # 2. 准备CoT数据集
    cot_dataset = create_cot_dataset(sp, cot_data_path)
    print(f"CoT数据集已准备，共 {len(cot_dataset)} 条数据")

    # 3. 链式思考微调
    print("开始链式思考微调...")
    train(model, cot_dataset, epochs=COT_EPOCHS, lr=COT_LR)

    # 4. 保存微调后的模型
    torch.save(model.state_dict(), out_path)
    print(f"微调后的模型已保存: {out_path}")

if __name__ == "__main__":
    main()