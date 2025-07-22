import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 设置随机种子
torch.manual_seed(42)

# 模拟 10 个 token 的 embedding
vocab_size = 10
embedding_dim = 256
token_labels = ["apple", "banana", "car", "train", "dog", "cat", "teacher", "student", "king", "queen"]

embedding_layer = nn.Embedding(vocab_size, embedding_dim)
token_ids = torch.arange(vocab_size)
embeddings = embedding_layer(token_ids).detach().numpy()

# 使用 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

# 可视化
plt.figure(figsize=(8, 6))
for i, label in enumerate(token_labels):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.5, y + 0.5, label, fontsize=12)
plt.title("t-SNE 可视化: 10个token的初始embedding")
plt.grid(True)
plt.tight_layout()
plt.show()
