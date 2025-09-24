# path: scripts/predict_gpt_RoPE.py
"""
使用训练好的 GPT 模型（RoPE版本）生成文本
用法：
    python scripts/predict_gpt_RoPE.py workdir/spm_shuihu.model workdir/gpt_shuihu_RoPE.pth "林教头在东京"
"""

import sys
import torch
import sentencepiece as spm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 128

# ----------------- RoPE 模型定义开始 -----------------
import math
from torch import nn


def apply_rope(x, freqs):
    x_real, x_imag = x.float().view(*x.shape[:-1], -1, 2).unbind(-1)
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    freqs_real, freqs_imag = freqs.view(*freqs.shape[:-1], -1, 2).unbind(-1)
    x_rotated_real = x_real * freqs_real - x_imag * freqs_imag
    x_rotated_imag = x_real * freqs_imag + x_imag * freqs_imag
    x_rotated = torch.stack((x_rotated_real, x_rotated_imag), dim=-1).flatten(start_dim=-2)
    return x_rotated.type_as(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(attn_dropout)
        self.register_buffer("freqs", self._create_freqs_buffer())

    def _create_freqs_buffer(self):
        head_dim = self.head_dim
        pos = torch.arange(BLOCK_SIZE, dtype=torch.float32)
        dim = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (dim / head_dim))
        freqs = torch.outer(pos, inv_freq)
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1).flatten(-2)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        q = apply_rope(q, self.freqs[:T].to(q.device))
        k = apply_rope(k, self.freqs[:T].to(k.device))

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * 4, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLike(nn.Module):
    def __init__(self, vocab_size, block_size, n_layers=2, dim=128, num_heads=4):
        super().__init__()
        # 没有位置嵌入层
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


# ----------------- RoPE 模型定义结束 -----------------

@torch.no_grad()
def generate(model, sp, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
    idx = torch.tensor([sp.encode(prompt, out_type=int)], device=DEVICE)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model(idx_cond)[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -1e10
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)
    return sp.decode(idx[0].tolist())


def main():
    if len(sys.argv) < 4:
        print("用法: python scripts/predict_gpt_RoPE.py 分词器模型 已训练模型 输入提示")
        sys.exit(1)

    sp_model, model_path, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    sp = spm.SentencePieceProcessor(model_file=sp_model)

    vocab_size = sp.get_piece_size()
    # 实例化 RoPE 版本的模型
    model = GPTLike(vocab_size, BLOCK_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    result = generate(model, sp, prompt)
    print("=== 输入提示 ===")
    print(prompt)
    print("=== 生成结果 ===")
    print(result)


if __name__ == "__main__":
    main()