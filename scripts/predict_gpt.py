# path: scripts/predict_gpt.py
"""
使用训练好的 GPT 模型生成文本
用法：
    python scripts/predict_gpt.py workdir/spm_shuihu.model workdir/gpt_shuihu.pth "宋江在梁山泊"
"""

import sys
import torch
import sentencepiece as spm
from train_gpt import GPTLike, DEVICE, BLOCK_SIZE

@torch.no_grad()
def generate(model, sp, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
    idx = torch.tensor([sp.encode(prompt, out_type=int)], device=DEVICE)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model(idx_cond)[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -1e10
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)
    return sp.decode(idx[0].tolist())

def main():
    if len(sys.argv) < 4:
        print("用法: python scripts/predict_gpt.py 分词器模型 已训练模型 输入提示")
        sys.exit(1)

    sp_model, model_path, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    sp = spm.SentencePieceProcessor(model_file=sp_model)

    vocab_size = sp.get_piece_size()
    model = GPTLike(vocab_size, BLOCK_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    result = generate(model, sp, prompt)
    print("=== 输入提示 ===")
    print(prompt)
    print("=== 生成结果 ===")
    print(result)

if __name__ == "__main__":
    main()
