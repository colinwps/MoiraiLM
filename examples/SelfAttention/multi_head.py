import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====== 1. 准备水浒传真实语料 ======
raw_texts = [
    "話說大宋仁宗天子在位，嘉祐三年三月三日五更三點，天子駕坐紫宸殿，受百官朝賀。但見：祥雲迷鳳閣，瑞氣罩龍樓。含煙御柳拂旌旗，帶露宮花迎劍戟。天香影裏，玉簪珠履聚丹墀。仙樂聲中，繡襖錦衣扶御駕。珍珠廉卷，黃金殿上現金輿。鳳尾扇開，白玉階前停寶輦。隱隱凈鞭三下響，層層文武兩班齊。",
    "那高俅在臨淮州，因得了赦宥罪犯，思鄉要回東京。這柳世權卻和東京城里金梁橋下開生藥鋪的董將士是親戚，寫了一封書札，收拾些人事盤纏，赍發高俅回東京，投奔董將士家過活。",
    "話說當時史進道：「卻怎生是好？」朱武等三個頭領跪下答道：「哥哥，你是乾淨的人，休為我等連累了大郎。可把索來綁縛我三個，出去請賞，免得負累了你不好看。」"
]

# ====== 2. 按字切分 ======
def char_tokenize(text):
    return [ch for ch in text if ch.strip()]  # 去掉空格、换行

sentences = [char_tokenize(t) for t in raw_texts]

# 构建词表
vocab = {}
for sent in sentences:
    for ch in sent:
        if ch not in vocab:
            vocab[ch] = len(vocab)

# ====== 3. 转成索引形式并做 padding ======
max_len = max(len(s) for s in sentences)
PAD_TOKEN = "<PAD>"
vocab[PAD_TOKEN] = len(vocab)

input_ids = []
for sent in sentences:
    ids = [vocab[ch] for ch in sent]
    # padding
    ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    input_ids.append(ids)

input_ids = torch.tensor(input_ids)  # (batch_size, seq_len)

# ====== 4. 多头自注意力模块 ======
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = dropout
        self.last_attn_weights = None  # 保存最后一次注意力权重 (batch, heads, seq, seq)

    def forward(self, x):
        B, T, C = x.size()

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        self.last_attn_weights = attn_weights.detach()  # (B, heads, T, T)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out

# ====== 5. 模型训练 ======
embed_dim = 32
num_heads = 4
vocab_size = len(vocab)

embedding = nn.Embedding(vocab_size, embed_dim)
model = MultiHeadSelfAttention(embed_dim, num_heads)
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

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.6f}")

# ====== 6. 可视化注意力热图 ======
for idx, sent in enumerate(sentences):
    attn = model.last_attn_weights[idx]  # (heads, seq, seq)
    sent_len = len(sent)
    for head in range(num_heads):
        plt.figure(figsize=(8, 6))
        plt.imshow(attn[head, :sent_len, :sent_len].numpy(), cmap='viridis')
        plt.title(f"第{idx+1}句 第{head+1}头 注意力矩阵")
        plt.xticks(ticks=np.arange(sent_len), labels=sent, rotation=90)
        plt.yticks(ticks=np.arange(sent_len), labels=sent)
        plt.xlabel("Key (字)")
        plt.ylabel("Query (字)")
        plt.colorbar(label="Attention Strength")
        for i in range(sent_len):
            for j in range(sent_len):
                plt.text(j, i, f"{attn[head, i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)
        plt.tight_layout()
        plt.savefig(f"attention_sentence{idx+1}_head{head+1}.png")
        plt.close()

print("注意力热图已保存。")
