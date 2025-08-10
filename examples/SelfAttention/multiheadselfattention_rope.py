# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import jieba
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 准备《水浒传》样本文本 =====
text_samples = [
    """张天师祈禳瘟疫，洪太尉误走妖魔。话说大宋天子仁宗皇帝在位年间，
       京师瘟疫流行，百姓多有染病。天子召张天师入宫祈禳，命洪太尉押送香火，
       不料误开封印，放出妖魔。""",
    """王教头私走延安府，九纹龙大闹史家村。史进自幼好武，学成十八般武艺，
       因打死恶霸，被官府缉拿。王进教头见势不妙，离开东京前往延安府，
       途经史家村。""",
    """史大郎夜走华阴县，鲁提辖拳打镇关西。史进与鲁达结义，路遇镇关西郑屠，
       见其欺压妇女，鲁达愤然出手，三拳打死郑屠，遂落草为寇。"""
]

# ===== 2. 中文分词 =====
def tokenize_texts(text_list):
    tokenized = []
    for t in text_list:
        words = list(jieba.cut(t))
        words = [w.strip() for w in words if w.strip()]
        tokenized.append(words)
    return tokenized

sentences = tokenize_texts(text_samples)

# ===== 3. 构建词表 =====
vocab = {}
for sent in sentences:
    for w in sent:
        if w not in vocab:
            vocab[w] = len(vocab)
vocab["<PAD>"] = len(vocab)

vocab_size = len(vocab)
embed_dim = 32
seq_len = max(len(s) for s in sentences)

# 将句子转为索引，并pad
def encode_sentences(sentences, vocab, seq_len):
    data = []
    for s in sentences:
        idxs = [vocab[w] for w in s]
        if len(idxs) < seq_len:
            idxs += [vocab["<PAD>"]] * (seq_len - len(idxs))
        data.append(idxs)
    return torch.tensor(data)

input_ids = encode_sentences(sentences, vocab, seq_len)

# ===== 4. RoPE实现 =====
def apply_rope(x):
    """
    支持输入维度:
      - (B, T, D)  或
      - (B, T, H, D)
    返回相同形状，且对最后一维做 RoPE（要求 D 为偶数）
    """
    orig_shape = x.shape
    if len(orig_shape) == 3:
        # (B, T, D) -> 转为 (B, T, 1, D) 方便统一处理
        x = x.unsqueeze(2)
        squeezed = True
    else:
        squeezed = False
        # 形状为 (B, T, H, D)
    # 现在 x.shape = (B, T, H, D)
    bsz, seqlen, nheads, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    device = x.device
    dtype = x.dtype

    half = head_dim // 2
    # theta: (half,)
    theta = 10000 ** (-torch.arange(0, half, device=device, dtype=dtype) / half)  # (half,)
    # seq positions: (seqlen,)
    seq_idx = torch.arange(seqlen, device=device, dtype=dtype)  # (seqlen,)
    # freqs: (seqlen, half)
    freqs = torch.einsum('n,d->nd', seq_idx, theta)

    cos = freqs.cos().view(1, seqlen, 1, half)  # (1, T, 1, half)
    sin = freqs.sin().view(1, seqlen, 1, half)  # (1, T, 1, half)

    x1 = x[..., :half]  # (B, T, H, half)
    x2 = x[..., half:]  # (B, T, H, half)

    x_rotated = torch.cat([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], dim=-1)  # (B, T, H, D)

    if squeezed:
        x_rotated = x_rotated.squeeze(2)  # back to (B, T, D)

    return x_rotated


# ===== 5. 多头注意力 with RoPE =====
class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.last_attn_weights = None

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # 应用 RoPE
        q = apply_rope(q)
        k = apply_rope(k)

        # 注意力计算
        attn_scores = torch.einsum('bthd,bshd->bhts', q, k) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        self.last_attn_weights = attn_weights.detach()

        out = torch.einsum('bhts,bshd->bthd', attn_weights, v)
        out = out.reshape(B, T, C)
        return self.out_proj(out)

# ===== 6. 模型训练 =====
embedding = nn.Embedding(vocab_size, embed_dim)
model = MultiHeadSelfAttentionRoPE(embed_dim, num_heads=4, dropout=0.1)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=1e-3)

epochs = 200
for epoch in range(epochs):
    model.train()
    x = embedding(input_ids)
    target = x.clone()
    out = model(x)
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# ===== 7. 注意力热图可视化 =====
def plot_attention(attn, sentence_tokens, filename):
    heads = attn.shape[0]
    fig, axes = plt.subplots(1, heads, figsize=(4*heads, 4))
    if heads == 1:
        axes = [axes]
    for h in range(heads):
        ax = axes[h]
        attn_head = attn[h].numpy()
        im = ax.imshow(attn_head, cmap='viridis')
        ax.set_xticks(np.arange(len(sentence_tokens)))
        ax.set_yticks(np.arange(len(sentence_tokens)))
        ax.set_xticklabels(sentence_tokens, rotation=90)
        ax.set_yticklabels(sentence_tokens)
        ax.set_title(f"Head {h+1}")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

model.eval()
with torch.no_grad():
    x = embedding(input_ids)
    _ = model(x)
    attn_weights = model.last_attn_weights  # (batch, heads, seq, seq)

    for i, tokens in enumerate(sentences):
        attn = attn_weights[i]
        plot_attention(attn.cpu(), tokens, f"rope_attention_sentence{i+1}.png")

print("RoPE多头注意力热图已生成，文件名为 rope_attention_sentenceX.png")
