# path: tools/clean_shuihu.py
"""
清洗《水浒传》原始文本，生成适合训练的纯净语料
会自动按中文标点（。！？）切分成句子，每句一行
用法：
    python tools/clean_shuihu.py data/raw_shuihu.txt data/shuihu.txt
"""

import sys
import re

def clean_text(text: str) -> str:
    # 只保留中文和常见标点
    allowed = re.compile(r"[^\u4e00-\u9fff。，、！？：；（）《》——…\n ]+")
    text = allowed.sub(" ", text)

    # 去掉多余空格
    text = re.sub(r"\s+", " ", text)

    # 去掉章节标题 “第X回 …”
    text = re.sub(r"第[一二三四五六七八九十百千0-9]+回.*\n", "", text)

    # 按标点分句（句号、问号、感叹号），切分后加回标点
    sentences = re.split(r"([。！？])", text)
    merged = []
    for i in range(0, len(sentences)-1, 2):
        s = sentences[i].strip()
        p = sentences[i+1].strip()
        if s:
            merged.append(s + p)

    # 去掉空白句子，避免太短
    merged = [s for s in merged if len(s) > 1]

    return "\n".join(merged)

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("用法: python tools/clean_shuihu.py 输入文件 [输出文件]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) == 3 else "data/shuihu.txt"

    with open(in_path, "r", encoding="utf-8") as f:
        raw = f.read()

    clean = clean_text(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(clean)

    print(f"清洗完成，输出保存到 {out_path}")
    print(f"示例前5行：\n" + "\n".join(clean.splitlines()[:5]))

if __name__ == "__main__":
    main()