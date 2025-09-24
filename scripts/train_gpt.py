# path: scripts/train_gpt.py
"""
训练简化版 GPT 模型
用法：
    python scripts/train_gpt.py workdir/spm_shuihu.model data/shuihu.txt workdir/gpt_shuihu.pth
"""

import sys
import re
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4


def clean_text(text: str) -> str:
    allowed = re.compile(r"[^\u4e00-\u9fff。，、！？：；（）《》——…\n ]+")
    text = allowed.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TextDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1: idx + 1 + self.block_size], dtype=torch.long)
        return x, y


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
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
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


def train(model, dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    step = 0
    for ep in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {ep + 1}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            step += 1
            if step % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / len(loader)
        ppl = math.exp(avg_loss)
        print(f"[Epoch {ep + 1}] Avg Loss {avg_loss:.4f} | PPL {ppl:.2f}")


def main():
    if len(sys.argv) < 4:
        print("用法: python scripts/train_gpt.py 分词器模型 输入语料 输出模型")
        sys.exit(1)

    sp_model, corpus_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sp = spm.SentencePieceProcessor(model_file=sp_model)

    with open(corpus_path, encoding="utf-8") as f:
        text = clean_text(f.read())
    ids = sp.encode(text, out_type=int)

    dataset = TextDataset(ids, BLOCK_SIZE)
    model = GPTLike(sp.get_piece_size(), BLOCK_SIZE).to(DEVICE)
    train(model, dataset)
    torch.save(model.state_dict(), out_path)
    print(f"✅ 模型已保存: {out_path}")


if __name__ == "__main__":
    main()