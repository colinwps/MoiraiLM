MoiraiLM 🌟
 
MoiraiLM 是一个开源框架，用于从零构建大型语言模型 (LLM) 🚀。它帮助开发者深入理解和掌握 LLM 的核心组件，包括分词器、Transformer 架构、训练和推理，通过最小化和干净的实现来实现。该名称“Moirai”源自希腊神话中三位命运女神，象征着对语言流动和结果的掌控，类似于 LLM 的生成能力。
English Version 🇬🇧
Project Overview 📖
MoiraiLM is an open-source framework for building large language models (LLMs) from scratch. It helps developers understand and master the core components of LLMs, including tokenizer, transformer architecture, training, and inference, through minimal and clean implementations. The name "Moirai" is inspired by the three goddesses of fate in Greek mythology, symbolizing control over the flow and outcome of language, much like LLMs.
Features ✨

✍️ Handwritten implementation of all LLM components from scratch.
🌐 Custom BPE tokenizer supporting English and Chinese text.
🛠️ Clean transformer code with support for RoPE (Rotary Position Embedding) and RMSNorm.
📚 Minimal training and inference loops designed for educational purposes.
🔗 Full-stack design covering tokenizer, model, training, and inference.

Project Structure 📂
The repository is organized as follows:
MoiraiLM/
├── tokenizer/         # BPE tokenizer implementation 🗣️
├── model/             # Transformer architecture 🧠
├── data/              # Data loading & preprocessing 📊
├── train/             # Training scripts 🏋️
├── inference/         # Text generation 📝
├── utils/             # Helpers and config files 🛠️
├── examples/          # Usage examples and notebooks 📓
├── requirements.txt   # Dependencies 📦
├── configs/           # Configuration files (e.g., train_config.yaml) ⚙️
└── README.md

Getting Started 🚀
1. Clone the Repository 📥
git clone https://github.com/colinwps/MoiraiLM.git
cd MoiraiLM

2. Install Dependencies 🛠️
pip install -r requirements.txt

(Note: Dependencies include PyTorch, NumPy, and other standard libraries for ML.)
3. Train BPE Tokenizer 🎓
Train the tokenizer on your corpus:
python tokenizer/train_bpe.py --input data/corpus.txt --vocab_size 5000


data/corpus.txt: Your training text corpus (supports mixed English/Chinese).
vocab_size: Desired vocabulary size (e.g., 5000).

4. Train the Model 🏋️
python train/train_lm.py --config configs/train_config.yaml


Customize hyperparameters in configs/train_config.yaml (e.g., batch size, learning rate, model dimensions).

5. Run Inference 📝
Generate text based on a prompt:
python inference/generate.py --prompt "Today is a beautiful day"

Example outputs:

Input: "Today is a beautiful day" → Output: ", perfect for a walk in the park." 🌳
Input: "The capital of China is" → Output: "Beijing." 🇨🇳

Examples 📓
Check the examples/ directory for Jupyter notebooks demonstrating:

Tokenizer training on custom data 🗣️
Fine-tuning the model on specific datasets 🔧
Advanced inference techniques like beam search 🔍

Contributing 🤝
We welcome contributions! 

🐛 Report bugs or request features via Issues.
💡 Submit pull requests for improvements.
⭐ Star the repo to show your support.

License 📜
This project is licensed under the MIT License - see the LICENSE file for details.
Contact 📬

GitHub: @colinwps
WeChat Official Account: 写代码的中年人 (The Middle-Aged Programmer)

"Destiny is not written — it’s generated, token by token." ✨

中文版 🇨🇳
项目概述 📖
MoiraiLM 是一个开源框架，用于从零构建大型语言模型 (LLM) 🚀。它帮助开发者深入理解和掌握 LLM 的核心组件，包括分词器、Transformer 架构、训练和推理，通过最小化和干净的实现。该名称“Moirai”源自希腊神话中三位命运女神，象征着对语言流动和结果的掌控，类似于 LLM 的生成能力。
特性 ✨

✍️ 从零手写实现所有 LLM 组件。
🌐 自定义 BPE 分词器，支持英文和中文文本。
🛠️ 干净的 Transformer 代码，支持 RoPE (旋转位置嵌入) 和 RMSNorm。
📚 最小化的训练和推理循环，专为教育目的设计。
🔗 全栈设计，涵盖分词器、模型、训练和推理。

项目结构 📂
仓库结构如下：
MoiraiLM/
├── tokenizer/         # BPE 分词器实现 🗣️
├── model/             # Transformer 架构 🧠
├── data/              # 数据加载与预处理 📊
├── train/             # 训练脚本 🏋️
├── inference/         # 文本生成 📝
├── utils/             # 辅助工具和配置文件 🛠️
├── examples/          # 使用示例和笔记本 📓
├── requirements.txt   # 依赖项 📦
├── configs/           # 配置文件 (例如 train_config.yaml) ⚙️
└── README.md

快速开始 🚀
1. 克隆仓库 📥
git clone https://github.com/colinwps/MoiraiLM.git
cd MoiraiLM

2. 安装依赖 🛠️
pip install -r requirements.txt

(注意：依赖包括 PyTorch、NumPy 等机器学习标准库。)
3. 训练 BPE 分词器 🎓
在语料库上训练分词器：
python tokenizer/train_bpe.py --input data/corpus.txt --vocab_size 5000


data/corpus.txt：您的训练文本语料库（支持中英文混合）。
vocab_size：期望的词汇表大小（例如 5000）。

4. 训练模型 🏋️
python train/train_lm.py --config configs/train_config.yaml


在 configs/train_config.yaml 中自定义超参数（例如批次大小、学习率、模型维度）。

5. 运行推理 📝
基于提示生成文本：
python inference/generate.py --prompt "今天天气真好"

示例输出：

输入："今天天气真好" → 输出："，适合出去散步。" 🌳
输入："今天天气好冷" → 输出："那应该要加衣服了" 🇨🇳

示例 📓
查看 examples/ 目录中的 Jupyter 笔记本，演示：

在自定义数据上训练分词器 🗣️
在特定数据集上微调模型 🔧
高级推理技术，如束搜索 🔍

贡献 🤝
欢迎贡献！

🐛 通过 Issues 报告 bug 或请求功能。
💡 提交拉取请求以改进代码。
⭐ 给仓库加星以支持开发。

许可 📜
本项目采用 MIT 许可 - 详见 LICENSE 文件。
联系方式 📬

GitHub: @colinwps
微信公众号：写代码的中年人

“命运并非书写而成——它是由一个一个 token 生成的。” ✨