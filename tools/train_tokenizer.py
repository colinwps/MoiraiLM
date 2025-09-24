<<<<<<< Updated upstream
# path: tools/train_tokenizer.py
"""
基于 SentencePiece 训练分词器 (BPE)
用法：
    python tools/train_tokenizer.py data/shuihu.txt workdir/spm_shuihu 8000
"""

import sys
import os
import sentencepiece as spm

def train_spm(input_path: str, model_prefix: str, vocab_size: int = 8000):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"语料文件不存在: {input_path}")

    print(f"开始训练分词器: {input_path}")
    spm.SentencePieceTrainer.Train(
        f"--input={input_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage=0.9995 "
        f"--bos_id=1 --eos_id=2 --unk_id=3"
    )
    print(f"分词器已保存: {model_prefix}.model / {model_prefix}.vocab")

def main():
    if len(sys.argv) < 3:
        print("用法: python tools/train_tokenizer.py 输入文本 模型前缀 [词表大小]")
        sys.exit(1)

    input_path = sys.argv[1]
    model_prefix = sys.argv[2]
    vocab_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 8000

    train_spm(input_path, model_prefix, vocab_size)

if __name__ == "__main__":
    main()
=======
# path: scripts/train_tokenizer.py
"""
使用SentencePiece训练分词器模型
用法：
    python scripts/train_tokenizer.py data/shuihu.txt workdir/spm_shuihu.model 32000
"""
import sys
import sentencepiece as spm


def train_spm_model(input_file, model_prefix, vocab_size):
    """
    训练一个SentencePiece分词器模型

    参数:
    input_file (str): 训练语料的路径。
    model_prefix (str): 训练模型文件的输出前缀。
    vocab_size (int): 词汇表大小。
    """
    # SentencePiece 训练参数
    # model_type: 可以是 'bpe', 'unigram', 'word' 或 'char'
    # 'bpe' 和 'unigram' 是最常用的子词分词类型
    # character_coverage: 保持训练语料中多少百分比的字符被覆盖
    # model_prefix: 输出文件的名字前缀 (例如: my_model.model, my_model.vocab)
    spm.SentencePieceTrainer.train(
        f'--input={input_file} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        '--model_type=bpe '
        '--character_coverage=0.9995 '
        '--num_threads=8 '
        '--bos_id=0 --eos_id=1 --unk_id=2 --pad_id=-1 '
    )


def main():
    if len(sys.argv) < 4:
        print("用法: python scripts/train_tokenizer.py <输入语料> <输出模型前缀> <词汇表大小>")
        sys.exit(1)

    input_file = sys.argv[1]
    model_prefix = sys.argv[2]
    vocab_size = int(sys.argv[3])

    print(f"  开始训练SentencePiece分词器...")
    print(f"   输入语料: {input_file}")
    print(f"   输出模型: {model_prefix}.model")
    print(f"   词汇表大小: {vocab_size}")

    train_spm_model(input_file, model_prefix, vocab_size)
    print("分词器模型训练完成！")


if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
