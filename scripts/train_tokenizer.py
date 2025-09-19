# path: scripts/train_tokenizer.py
"""
训练 SentencePiece 分词器 (BPE)
用法：
    python scripts/train_tokenizer.py data/shuihu.txt workdir/spm_shuihu 8000
"""

import sys
import os
import sentencepiece as spm

def train_tokenizer(input_path: str, model_prefix: str, vocab_size: int = 8000):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"语料文件不存在: {input_path}")

    print(f"[INFO] 开始训练分词器: {input_path}")
    spm.SentencePieceTrainer.Train(
        f"--input={input_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage=0.9995 "
        f"--bos_id=1 --eos_id=2 --unk_id=3"
    )
    print(f"[INFO] 分词器已保存: {model_prefix}.model / {model_prefix}.vocab")

def main():
    if len(sys.argv) < 3:
        print("用法: python scripts/train_tokenizer.py 输入文本 模型前缀 [词表大小]")
        sys.exit(1)

    input_path = sys.argv[1]
    model_prefix = sys.argv[2]
    vocab_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 8000

    train_tokenizer(input_path, model_prefix, vocab_size)

if __name__ == "__main__":
    main()
