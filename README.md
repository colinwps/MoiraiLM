# MoiraiLM

🚀 **MoiraiLM** 是一个从零实现大语言模型（LLM, Large Language Model）的开源项目，旨在帮助开发者深入理解大模型的构建原理、核心模块、训练流程以及推理机制。

项目以教学友好、模块清晰为目标，面向希望掌握 Transformer 架构及其在语言建模中应用的 AI 开发者、研究者和爱好者。

---

## 🧠 项目目标

- 从零开始搭建一个完整的 LLM 框架；
- 不依赖复杂的第三方黑盒模块，代码可读性强；
- 覆盖 tokenizer、embedding、attention、position encoding、optimizer、训练、推理等核心环节；
- 支持从小规模模型到多层 Transformer 架构的扩展。

---

## 📦 项目结构

```bash
MoiraiLM/
├── tokenizer/         # 实现 BPE 分词器及训练、保存、加载逻辑
├── model/             # 模型结构，包括 Transformer、注意力机制等
├── data/              # 文本数据预处理与加载
├── train/             # 模型训练脚本与训练配置
├── inference/         # 推理代码，支持基础生成能力
├── utils/             # 辅助工具函数，如配置管理、日志等
├── examples/          # 示例代码，快速上手训练与推理
└── README.md          # 项目说明文档
