import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------- 定义多头自注意力模块 -------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # 查询、键、值的线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = dropout
        self.last_attn_weights = None  # 保存最近一次的注意力权重（每头一个）

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seq_len, embed_dim

        # 线性变换：计算查询、键、值
        Q = self.q_proj(x)  # (B, T, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 拆分为多头
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, T, T)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 保存注意力权重（用于可视化）
        self.last_attn_weights = attn_weights.detach()  # (B, num_heads, T, T)

        # 计算注意力输出
        out = torch.matmul(attn_weights, V)  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)  # (B, T, embed_dim)

        # 输出投影
        out = self.out_proj(out)
        return out

# ------- 创建模拟数据集 -------
# 模拟一个小型词汇表和词嵌入
vocab = {"写": 0, "代码": 1, "的": 2, "中年人": 3, "天天": 4, "<PAD>": 5}
embed_dim = 16
num_heads = 4  # 多头注意力的头数
vocab_size = len(vocab)
embedding = nn.Embedding(vocab_size, embed_dim)  # 随机初始化词嵌入

# 句子数据
sentences = [
    ["写", "代码", "的", "中年人"],
    ["天天", "写", "代码", "<PAD>"]
]
batch_size = len(sentences)
seq_len = len(sentences[0])  # 统一序列长度为4

# 将句子转换为索引
input_ids = torch.tensor([[vocab[word] for word in sent] for sent in sentences])  # (batch_size, seq_len)

# ------- 参数设置 -------
epochs = 200
dropout = 0.1
model = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------- 训练模型 -------
for epoch in range(epochs):
    model.train()
    # 每次循环重新计算输入以创建新的计算图
    x = embedding(input_ids)  # (batch_size, seq_len, embed_dim)
    target = x.clone()  # 目标是输入的复制

    out = model(x)
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.6f}")

# ------- 可视化多头注意力权重 -------
# 为每个句子可视化每个头的注意力矩阵
for sent_idx, sentence in enumerate(sentences):
    for head_idx in range(num_heads):
        attention = model.last_attn_weights[sent_idx, head_idx].numpy()  # (seq_len, seq_len)
        plt.figure(figsize=(8, 6))
        plt.imshow(attention, cmap='viridis')
        plt.title(f"Attention Matrix (Head {head_idx+1}) for Sentence: {' '.join(sentence)}")
        plt.xticks(ticks=np.arange(seq_len), labels=sentence)
        plt.yticks(ticks=np.arange(seq_len), labels=sentence)
        plt.xlabel("Key (Word)")
        plt.ylabel("Query (Word)")
        plt.colorbar(label="Attention Strength")
        for i in range(seq_len):
            for j in range(seq_len):
                plt.text(j, i, f"{attention[i,j]:.2f}", ha="center", va="center", color="white")
        plt.tight_layout()
        plt.savefig(f"attention_matrix_sentence{sent_idx+1}_head{head_idx+1}.png")
        plt.show()