import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import sentencepiece as spm


# ===== Transformer 模型结构（完整复制自训练代码） =====
class TransformerInputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout=0.1):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.embedding = TransformerInputEmbedding(vocab_size, embed_dim, max_seq_len)
        self.transformer = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids, mask=None):
        embeds = self.embedding(token_ids)
        transformer_output = self.transformer(embeds, mask)
        logits = self.output_layer(transformer_output)
        return logits


# ===== 推理函数与主程序 =====
def load_tokenizer(model_file='shuihu_bpe.model'):
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp


def load_trained_model(model_path, vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, device):
    model = LanguageModel(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_seq_len, gen_len=50, temperature=1.0, device='cpu'):
    vocab_size = model.output_layer.out_features
    input_ids = tokenizer.encode(prompt, out_type=int)
    input_ids = [i for i in input_ids if i < vocab_size][:max_seq_len]  # 修复：过滤非法 token id

    if not input_ids:
        raise ValueError("输入 prompt 编码后为空或所有 token 超出词表范围")

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    for _ in range(gen_len):
        seq_len = input_tensor.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor, mask)

        next_token_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
        if input_tensor.size(1) >= max_seq_len:
            break

    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids)


if __name__ == '__main__':
    prompt = '鲁智深'

    vocab_size = 7000
    embed_dim = 256
    max_seq_len = 50
    num_heads = 8
    ff_dim = 1024
    num_layers = 4
    model_path = 'shuihu_transformer.pt'
    device = torch.device('cpu')

    tokenizer = load_tokenizer('shuihu_bpe.model')
    model = load_trained_model(model_path, vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, device)

    generated_text = generate(model, tokenizer, prompt, max_seq_len, gen_len=50, temperature=1.0, device=device)
    print(f'输入: {prompt}\n生成: {generated_text}')