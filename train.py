import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
import math
import re
import os


# 1. BPE 分词
def train_bpe_tokenizer(corpus_file, vocab_size=7000, model_prefix='shuihu_bpe'):
    try:
        # Verify input file
        with open(corpus_file, 'r', encoding='utf-8') as f:
            text = f.read()
            if len(text.strip()) == 0:
                raise ValueError("Input file is empty or contains only whitespace.")
            print(f"Input file size: {len(text)} characters")

        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type='bpe'
        )
        print(f"BPE model trained successfully. Files: {model_prefix}.model, {model_prefix}.vocab")
        return spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
    except Exception as e:
        print(f"Error training BPE tokenizer: {e}")
        raise


# 2. 数据预处理（适配一章一行）
def prepare_data(text_file, sp, max_seq_len):
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        all_token_ids = []
        for line in lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 按句子分割长章节
            sentences = re.split(r'[。！？]', line)
            sentences = [s.strip() for s in sentences if s.strip()]

            # 对每个句子进行分词
            for sentence in sentences:
                token_ids = sp.encode(sentence, out_type=int)
                all_token_ids.extend(token_ids)

        # 分割为固定长度序列
        data = []
        for i in range(0, len(all_token_ids) - max_seq_len, 1):
            data.append(all_token_ids[i:i + max_seq_len + 1])

        if not data:
            raise ValueError("No valid sequences generated. Check input data or max_seq_len.")
        print(f"Generated {len(data)} sequences of length {max_seq_len + 1}")
        return data
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        raise


# 3. 数据加载器
def create_data_loader(token_ids, seq_len, batch_size):
    data = torch.tensor(token_ids, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(data[:, :-1], data[:, 1:])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 4. Transformer 模型（简化为展示，完整实现与之前相同）
class TransformerInputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout=0.1):
        super(TransformerInputEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.size()
        token_embeds = self.token_embedding(token_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_linear(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers):
        super(LanguageModel, self).__init__()
        self.embedding = TransformerInputEmbedding(vocab_size, embed_dim, max_seq_len)
        self.transformer = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids, mask=None):
        embeds = self.embedding(token_ids)
        transformer_output = self.transformer(embeds, mask)
        logits = self.output_layer(transformer_output)
        return logits


# 5. 训练函数
def train_model(model, data_loader, epochs, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            seq_len = inputs.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(device)

            optimizer.zero_grad()
            logits = model(inputs, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')


# 6. 主程序
def main():
    # 参数设置
    corpus_file = 'data/raw_shuihu.txt'
    vocab_size = 7000  # Reduced to avoid vocab size error
    embed_dim = 256
    max_seq_len = 50
    num_heads = 8
    ff_dim = 1024
    num_layers = 4
    batch_size = 32
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练 BPE 分词器
    sp = train_bpe_tokenizer(corpus_file, vocab_size)

    # 准备数据
    token_ids = prepare_data(corpus_file, sp, max_seq_len)

    # 创建数据加载器
    data_loader = create_data_loader(token_ids, max_seq_len, batch_size)

    # 初始化模型
    model = LanguageModel(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)

    # 训练模型
    train_model(model, data_loader, epochs, device)

    # 保存模型
    torch.save(model.state_dict(), 'shuihu_transformer.pt')


if __name__ == '__main__':
    main()