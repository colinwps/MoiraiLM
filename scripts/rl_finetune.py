# path: scripts/rl_finetune.py
"""
使用简化的强化学习进行微调
（请注意，这是一个简化的框架，不包含完整PPO）
"""

import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from train_gpt import GPTLike, DEVICE, BLOCK_SIZE
import sentencepiece as spm
from train_rm import RewardModel  # 导入奖励模型

# RL参数
RL_EPOCHS = 1
RL_LR = 1e-6


# 这是一个简化的PPO训练函数
def rl_train(policy_model, reward_model, sp, prompt_list):
    opt = torch.optim.AdamW(policy_model.parameters(), lr=RL_LR)

    # 冻结奖励模型，只训练策略模型
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    policy_model.train()

    for epoch in range(RL_EPOCHS):
        pbar = tqdm(prompt_list, desc=f"RL Epoch {epoch + 1}")
        for prompt in pbar:
            prompt_ids = torch.tensor([sp.encode(prompt, out_type=int)], device=DEVICE)

            # 使用策略模型生成回答
            generated_ids = generate_for_rl(policy_model, prompt_ids, sp.eos_id())

            # 使用奖励模型对生成的回答打分
            with torch.no_grad():
                reward = reward_model(generated_ids).item()

            # 简化版：使用奖励作为损失，直接反向传播
            # 实际PPO过程要复杂得多，涉及价值函数和KL散度惩罚
            logits = policy_model(generated_ids)[:, :-1, :]
            targets = generated_ids[:, 1:]

            # 计算生成路径的对数概率
            log_probs = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            loss = (-log_probs.sum() * reward).mean()  # 奖励越高，损失越小

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(reward=f"{reward:.2f}", loss=f"{loss.item():.4f}")


@torch.no_grad()
def generate_for_rl(model, prompt_ids, eos_id, max_len=50):
    # 简化的生成函数，不使用采样
    idx = prompt_ids
    for _ in range(max_len):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model(idx_cond)[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == eos_id:
            break
    return idx


def main():
    if len(sys.argv) < 5:
        print("用法: python scripts/rl_finetune.py 分词器模型 SFT模型 RM模型 RL数据 输出模型")
        sys.exit(1)

    sp_model_path, sft_model_path, rm_model_path, rl_data_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], \
    sys.argv[4], sys.argv[5]

    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    vocab_size = sp.get_piece_size()

    # 1. 加载SFT模型（作为策略模型）
    policy_model = GPTLike(vocab_size, BLOCK_SIZE).to(DEVICE)
    policy_model.load_state_dict(torch.load(sft_model_path, map_location=DEVICE))

    # 2. 加载奖励模型
    reward_model = RewardModel(vocab_size, BLOCK_SIZE).to(DEVICE)
    reward_model.load_state_dict(torch.load(rm_model_path, map_location=DEVICE))

    # 3. 准备RL训练数据（用户问题列表）
    with open(rl_data_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f]
    print(f"RL数据已准备，共 {len(prompts)} 个提示")

    # 4. 运行RL微调
    print("开始RL微调...")
    rl_train(policy_model, reward_model, sp, prompts)

    torch.save(policy_model.state_dict(), out_path)
    print(f"RL微调后的模型已保存: {out_path}")


if __name__ == "__main__":
    main()