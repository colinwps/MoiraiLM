import json
from collections import defaultdict
import re
import os

class BPETokenizer:
    def __init__(self):
        self.vocab = {}  # token -> id
        self.inverse_vocab = {}  # id -> token
        self.merges = []  # List of (token1, token2) pairs
        self.merge_ranks = {}  # pair -> rank
        self.next_id = 0
        self.special_tokens = []

    def get_stats(self, word_freq):
        pairs = defaultdict(int)
        for word, freq in word_freq.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, word_freq):
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_word_freq = {}
        pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
        for word, freq in word_freq.items():
            new_word = pattern.sub(replacement, word)
            new_word_freq[new_word] = freq
        return new_word_freq

    def train(self, corpus, vocab_size, special_tokens=None):
        if special_tokens is None:
            special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        self.special_tokens = special_tokens

        for token in special_tokens:
            self.vocab[token] = self.next_id
            self.inverse_vocab[self.next_id] = token
            self.next_id += 1

        word_freq = defaultdict(int)
        for text in corpus:
            words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
            for word in words:
                word_freq[' '.join(list(word))] += 1

        while len(self.vocab) < vocab_size:
            pairs = self.get_stats(word_freq)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            word_freq = self.merge_vocab(best_pair, word_freq)
            new_token = ''.join(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = self.next_id
                self.inverse_vocab[self.next_id] = new_token
                self.next_id += 1

    def encode(self, text):
        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        token_ids = []
        for word in words:
            tokens = list(word)
            while len(tokens) > 1:
                pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                merge_pair = None
                merge_rank = float('inf')
                for pair in pairs:
                    rank = self.merge_ranks.get(pair, float('inf'))
                    if rank < merge_rank:
                        merge_pair = pair
                        merge_rank = rank
                if merge_pair is None:
                    break
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge_pair:
                        new_tokens.append(''.join(merge_pair))
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            for token in tokens:
                token_ids.append(self.vocab.get(token, self.vocab['[UNK]']))
        return token_ids

    def decode(self, token_ids):
        tokens = [self.inverse_vocab.get(id, '[UNK]') for id in token_ids]
        return ''.join(tokens)

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, 'merges.txt'), 'w', encoding='utf-8') as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
        with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
            config = {
                "model_type": "bpe",
                "vocab_size": len(self.vocab),
                "special_tokens": self.special_tokens,
                "merges_file": "merges.txt",
                "vocab_file": "vocab.json"
            }
            json.dump(config, f, ensure_ascii=False, indent=2)

    def export_token_map(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for token_id, token in self.inverse_vocab.items():
                f.write(f"{token_id}\t{token}\t{' '.join(token)}\n")

    def print_visualization(self, text):
        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        visualized = []
        for word in words:
            tokens = list(word)
            while len(tokens) > 1:
                pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                merge_pair = None
                merge_rank = float('inf')
                for pair in pairs:
                    rank = self.merge_ranks.get(pair, float('inf'))
                    if rank < merge_rank:
                        merge_pair = pair
                        merge_rank = rank
                if merge_pair is None:
                    break
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge_pair:
                        new_tokens.append(''.join(merge_pair))
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            visualized.append(' '.join(tokens))
        return ' | '.join(visualized)

    def load(self, path):
        with open(os.path.join(path, 'vocab.json'), 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            self.vocab = {k: int(v) for k, v in self.vocab.items()}
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.next_id = max(self.vocab.values()) + 1

        with open(os.path.join(path, 'merges.txt'), 'r', encoding='utf-8') as f:
            self.merges = []
            self.merge_ranks = {}
            for i, line in enumerate(f):
                token1, token2 = line.strip().split()
                pair = (token1, token2)
                self.merges.append(pair)
                self.merge_ranks[pair] = i

        config_path = os.path.join(path, 'tokenizer_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.special_tokens = config.get("special_tokens", [])


def load_corpus_from_file(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


if __name__ == "__main__":
    corpus = load_corpus_from_file("data/raw_shuihu.txt")

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=500)

    test_text = "且说鲁智深自离了五台山文殊院，取路投东京来，行了半月之上"
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    tokenizer.save("./bpe_tokenizer")
    tokenizer.export_token_map("./bpe_tokenizer/token_map.tsv")

    print("\nSaved files:")
    print(f"vocab.json: {os.path.exists('./bpe_tokenizer/vocab.json')}")
    print(f"merges.txt: {os.path.exists('./bpe_tokenizer/merges.txt')}")
    print(f"tokenizer_config.json: {os.path.exists('./bpe_tokenizer/tokenizer_config.json')}")
    print(f"token_map.tsv: {os.path.exists('./bpe_tokenizer/token_map.tsv')}")

    print("\nVisualization:")
    print(tokenizer.print_visualization(test_text))
