
# MoiraiLM

**Build your own destiny, token by token.**  
**用代码编织语言的命运。**

---

## 🧬 What is MoiraiLM? / 什么是 MoiraiLM？

**MoiraiLM** is an open-source framework for building large language models (LLMs) from scratch.  
It aims to help developers understand and master the core components of modern LLMs—tokenizer, transformer architecture, training, and inference—through minimal yet clean implementations.

**MoiraiLM 是一个从零实现大语言模型（LLM, Large Language Model）的开源项目。**  
它通过简洁清晰的实现，帮助开发者全面理解并掌握大模型的分词器、Transformer 架构、训练流程与推理机制。

> The name “Moirai” comes from Greek mythology—the three goddesses of fate who spin the thread of life.  
> Just like LLMs, they determine the flow and outcome of language.

> “Moirai” 来自希腊神话中的命运三女神，掌控人类的命运之线，  
> 正如语言模型主导文本生成的节奏与方向。

---

## 🔥 Features / 项目特色

| Feature (English)                                  | 特性（中文）                                  |
|---------------------------------------------------|----------------------------------------------|
| From-scratch implementation of LLM components     | 所有模块均为手写，从零构建大模型              |
| Custom BPE tokenizer supporting English and Chinese | 支持中英文本的 BPE 分词器                     |
| Clean transformer code with support for RoPE, RMSNorm | 支持 RoPE、RMSNorm 等模块，结构清晰          |
| Minimal training/inference loop for educational purpose | 简洁的训练与推理流程，适合学习与扩展          |
| Full-stack design: Tokenizer → Model → Training → Inference | 从数据预处理到模型生成的完整链路               |

---

## 📁 Project Structure / 项目结构

```bash
MoiraiLM/
├── tokenizer/         # BPE tokenizer
├── model/             # Transformer architecture
├── data/              # Data loading & preprocessing
├── train/             # Training scripts
├── inference/         # Text generation
├── utils/             # Helpers and config
├── examples/          # Usage examples and notebooks
└── README.md
```

---

## 🚀 Getting Started / 快速开始

### 1️⃣ Clone the repo / 克隆项目

```bash
git clone https://github.com/colinwps/MoiraiLM.git
cd MoiraiLM
```

### 2️⃣ Install dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

### 3️⃣ Train BPE tokenizer / 训练分词器

```bash
python tokenizer/train_bpe.py --input data/corpus.txt --vocab_size 5000
```

### 4️⃣ Train the model / 训练模型

```bash
python train/train_lm.py --config configs/train_config.yaml
```

### 5️⃣ Run inference / 推理生成文本

```bash
python inference/generate.py --prompt "今天天气真好"
```

---

## 💡 Example Output / 示例输出

| Prompt / 输入             | Output / 输出                         |
|--------------------------|--------------------------------------|
| 今天天气                 | 真好，适合出去走走。                  |
| 中国的首都是             | 北京。                                |
| ChatGPT 是                | 一个由 OpenAI 开发的大型语言模型。     |

---

## 📚 Recommended Readings / 推荐阅读

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT Papers by OpenAI](https://openai.com/research/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Byte Pair Encoding Tutorial](https://huggingface.co/course/chapter6/6)

---

## 🙌 Contributing / 贡献方式

欢迎任何形式的参与：

- 💬 提交 [Issue](https://github.com/colinwps/MoiraiLM/issues) 反馈问题
- 🔧 提交 PR 修复 bug 或改进功能
- ⭐️ Star 本项目以支持发展

Welcome contributions in any form:

- 💬 Open an issue for questions, ideas or bugs
- 🔧 Submit PRs to improve features
- ⭐️ Star the repo to support the project

---

## 📄 License / 许可证

This project is licensed under the MIT License.  
本项目基于 MIT 协议开源，欢迎自由使用与修改。

---

## 📬 Contact / 联系方式

- GitHub: [@colinwps](https://github.com/colinwps)
- WeChat Official Account / 微信公众号：**写代码的中年人**

---

> ✨ “Destiny is not written — it’s generated, token by token.”  
> ✨ “命运不是注定的，而是一个个 token 拼出来的。” —— MoiraiLM
