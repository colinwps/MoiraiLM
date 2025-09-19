import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------- 定义可训练的自注意力模块 -------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.last_attn_weights = None

    def forward(self, x):
        B, T, C = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        self.last_attn_weights = attn_weights.detach()

        out = torch.matmul(attn_weights, V)
        out = self.out_proj(out)
        return out

# ------- Create a Simulated Dataset -------
# Simulate a small vocabulary and word embeddings
vocab = {"写": 0, "代码": 1, "的": 2, "中年人": 3, "天天": 4, "<PAD>": 5}
embed_dim = 16
vocab_size = len(vocab)
embedding = nn.Embedding(vocab_size, embed_dim)  # Randomly initialized word embeddings

# Sentence data
sentences = [
    ["写", "代码", "的", "中年人"],
    ["天天", "写", "代码", "<PAD>"]  # Pad the second sentence to match length
]
batch_size = len(sentences)
seq_len = len(sentences[0])  # Sentences have the same length (4)

# Convert sentences to indices
input_ids = torch.tensor([[vocab[word] for word in sent] for sent in sentences])  # (batch_size, seq_len)

# ------- Parameter Settings -------
epochs = 200
dropout = 0.1
model = SelfAttention(embed_dim, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------- Train the Model -------
for epoch in range(epochs):
    model.train()
    # Compute input inside the loop to create a fresh computation graph
    x = embedding(input_ids)  # (batch_size, seq_len, embed_dim)
    target = x.clone()  # Target is the same as input for this task

    out = model(x)
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backward()  # Compute gradients
    optimizer.step()  # Update model parameters

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.6f}")

# ------- Visualize Attention Weights -------
# Visualize attention matrix for the first sentence
attention = model.last_attn_weights[0].numpy()  # (seq_len, seq_len)
sentence = sentences[0]  # ["写", "代码", "的", "中年人"]

plt.figure(figsize=(8, 6))
plt.imshow(attention, cmap='viridis')
plt.title(f"Attention Matrix for Sentence: {' '.join(sentence)}")
plt.xticks(ticks=np.arange(seq_len), labels=sentence)
plt.yticks(ticks=np.arange(seq_len), labels=sentence)
plt.xlabel("Key (Word)")
plt.ylabel("Query (Word)")
plt.colorbar(label="Attention Strength")
for i in range(seq_len):
    for j in range(seq_len):
        plt.text(j, i, f"{attention[i,j]:.2f}", ha="center", va="center", color="white")
plt.tight_layout()
plt.savefig("attention_matrix_sentence1.png")
plt.show()

# Visualize attention matrix for the second sentence
attention = model.last_attn_weights[1].numpy()
sentence = sentences[1]  # ["天天", "写", "代码", "<PAD>"]

plt.figure(figsize=(8, 6))
plt.imshow(attention, cmap='viridis')
plt.title(f"Attention Matrix for Sentence: {' '.join(sentence)}")
plt.xticks(ticks=np.arange(seq_len), labels=sentence)
plt.yticks(ticks=np.arange(seq_len), labels=sentence)
plt.xlabel("Key (Word)")
plt.ylabel("Query (Word)")
plt.colorbar(label="Attention Strength")
for i in range(seq_len):
    for j in range(seq_len):
        plt.text(j, i, f"{attention[i,j]:.2f}", ha="center", va="center", color="white")
plt.tight_layout()
plt.savefig("attention_matrix_sentence2.png")
plt.show()