# path: scripts/train_gpt_RoPE.py (优化版)
"""
训练简化版 GPT 模型，使用旋转位置编码（RoPE）
用法：
    python scripts/train_gpt_RoPE.py workdir/spm_wiki.model data/cleaned_wiki_full.txt workdir/gpt_100M_RoPE.pth
"""

import sys
import os
import re
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from tqdm import tqdm

# --- 全局常量和超参数 (修改后的) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 512  # 增加序列长度以捕捉长程依赖
BATCH_SIZE = 32  # 增加批次大小 (如果显存允许)
EPOCHS = 1  # 亿级参数模型预训练通常只需要 1-3 个 Epoch
LR = 3e-4
# 模型配置：瞄准 1 亿参数左右
MODEL_DIM = 768
N_LAYERS = 12
NUM_HEADS = 12
# ------------------------------------

# 启用混合精度训练和 TF32 (A100/H100/4090 等现代卡)
if DEVICE == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True


# ... clean_text 函数保持不变 ...
def clean_text(text: str) -> str:
    allowed = re.compile(r"[^\u4e00-\u9fff。，、！？：；（）《》——…\n ]+")
    text = allowed.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TextDataset(Dataset):
    def __init__(self, token_ids: list, block_size: int):
        self.ids = token_ids
        self.block_size = block_size

    def __len__(self):
        # 减去 block_size + 1 以确保能取出 x 和 y
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1: idx + 1 + self.block_size], dtype=torch.long)
        return x, y


# --- RoPE 旋转函数优化：使用 torch.complex ---
def apply_rope(x, freqs):
    # x: (B, T, H, D) -> (B*T*H, D)
    # freqs: (T, D)

    # 1. 转换为复数形式 (D/2 个复数)
    x_reshaped = x.float().view(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshaped)

    freqs_reshaped = freqs.view(-1, freqs.shape[-1] // 2)
    freqs_complex = torch.view_as_complex(freqs_reshaped)

    # 2. 广播并应用旋转 (点乘复数即可完成旋转)
    # T 维度通过广播对齐
    x_rotated_complex = x_complex * freqs_complex[:x.shape[1], None, :]

    # 3. 转换回实数形式并展平
    x_rotated_reshaped = torch.view_as_real(x_rotated_complex)
    x_rotated = x_rotated_reshaped.flatten(start_dim=-2).type_as(x)
    return x_rotated


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)  # 惯例不加偏置
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(attn_dropout)

        # 预计算 RoPE 频率
        self.register_buffer("freqs", self._create_freqs_buffer(block_size, self.head_dim))

    def _create_freqs_buffer(self, block_size, head_dim):
        pos = torch.arange(block_size, dtype=torch.float32)
        dim = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (dim / head_dim))
        freqs = torch.outer(pos, inv_freq)
        # 将 cos 和 sin 堆叠并展平 (用于 apply_rope 简化版)
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1).flatten(-2)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        # q, k, v: (B, H, T, D_head)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        # 应用 RoPE 到 q 和 k (维度需要转换为 B, T, H, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        q = apply_rope(q, self.freqs[:T].to(q.device))
        k = apply_rope(k, self.freqs[:T].to(k.device))

        # 重新转换回 (B, H, T, D_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # MatMul Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Causal Mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        out = att @ v  # (B, H, T, D_head)

        # 合并头并输出
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(dim)
        # FeedForward 维度通常设置为 4 * dim
        self.ff = FeedForward(dim, dim * 4, dropout)

    def forward(self, x):
        # 使用 Post-LN (GPT-style)
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLike(nn.Module):
    def __init__(self, vocab_size, block_size, n_layers=N_LAYERS, dim=MODEL_DIM, num_heads=NUM_HEADS):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, block_size)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)  # 惯例不加偏置
        self.block_size = block_size
        self.apply(self._init_weights)

        # 打印模型参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"模型参数量: {n_params / 1e6:.2f} M")  # 打印参数量，以 M (百万) 为单位

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 权重初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx):
        x = self.token_emb(idx)  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.head(x)


def train(model, dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)  # 增加 workers
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    step = 0

    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    for ep in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {ep + 1}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # 1. 前向传播 (在 autocast 区域内)
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))

            # 2. 反向传播和优化 (使用 GradScaler)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            step += 1
            if step % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        ppl = math.exp(avg_loss)
        print(f"[Epoch {ep + 1}] Avg Loss {avg_loss:.4f} | PPL {ppl:.2f}")


def main():
    if len(sys.argv) < 4:
        print("用法: python scripts/train_gpt_RoPE.py 分词器模型 输入语料 输出模型")
        sys.exit(1)

    sp_model, corpus_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sp = spm.SentencePieceProcessor(model_file=sp_model)

    # --- 数据加载优化：避免内存溢出 ---
    print("⏳ 正在加载和编码语料...")
    # 假设你的 cleaned_wiki_full.txt 是一个巨大的文件，我们一次性读取并编码，
    # 但在实际亿级数据训练中，你可能需要修改这里使用内存映射或分块读取。
    # 对于 2.35 亿 Token，一次性加载是可行的。

    with open(corpus_path, encoding="utf-8") as f:
        text = clean_text(f.read())
    ids = sp.encode(text, out_type=int)
    print(f"✅ 语料编码完成。总 Token 数: {len(ids)}")

    dataset = TextDataset(ids, BLOCK_SIZE)

    # 创建模型，使用新的高参数配置
    model = GPTLike(sp.get_piece_size(), BLOCK_SIZE).to(DEVICE)
    print(f"🚀 开始在 {DEVICE} 上训练模型...")

    train(model, dataset)

    # 确保在保存前将模型切换回 CPU 内存
    torch.save(model.state_dict(), out_path)
    print(f"模型已保存: {out_path}")


if __name__ == "__main__":
    main()