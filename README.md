# 待修改
hippo模型的输入向量，输出向量的位置调整。
模型的输入逻辑还有错误
把hippomodel的输入改为embedding后的向量

# HIPPO: 对话历史记忆融合系统

HIPPO (History-Integrated Processing for Personalized Outputs) 是一个结合对话历史记忆功能的大语言模型增强系统。该系统通过引入专门的记忆模型（HippoModel），使大语言模型能够更好地理解和利用对话历史信息，提供更加连贯和个性化的回答。

## 项目概述

HIPPO系统的核心创新点在于：
- 将对话历史信息通过专门的记忆模型处理，生成记忆表示
- 将记忆表示与大语言模型的输出特征进行融合，实现记忆增强
- 采用非侵入式设计，不需要修改大语言模型的底层结构

## 系统架构

```
┌─────────────────┐      ┌─────────────────┐
│  对话历史输入   │ ────> │  HippoModel    │
│  (dialog_history)│      │ (记忆编码器)   │
└─────────────────┘      └──────────┬──────┘
                                    │
                                    ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  当前查询输入   │ ────> │  基础大语言模型 │ ────> │    特征融合     │ ────> 增强输出
│  (memory_query) │      │ (ModifiedQwen)  │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.28+
- Datasets

### 安装依赖

```bash
pip install -r requirements.txt
```

### 生成示例数据

系统提供了生成示例训练数据的功能：

```bash
python data_processor.py
```

这将在当前目录生成`demo_training_data.jsonl`文件，包含100条示例对话数据。

### 模型训练

使用以下命令启动模型训练：

```bash
python train.py --model_name_or_path /path/to/qwen/model --data_path demo_training_data.jsonl --output_dir ./output --max_length 512 --batch_size 2
```

可选参数：
- `--model_name_or_path`: 预训练模型路径
- `--data_path`: 训练数据路径
- `--output_dir`: 模型保存路径
- `--max_length`: 最大序列长度
- `--batch_size`: 训练批次大小
- `--learning_rate`: 学习率
- `--num_train_epochs`: 训练轮数

### 推理使用

训练完成后，可以使用`inference.py`进行推理：

```bash
python inference.py --model_path ./output --dialog_history "用户: 你好
助手: 你好！我是AI助手，有什么可以帮助你的吗？" --query "你刚才说你是谁？"
```

## 文件结构

```
├── data_processor.py    # 数据加载和预处理模块
├── generate_data.py     # 数据生成工具
├── hippo_model.py       # 记忆编码模型实现
├── model_customization.py # 大模型定制和特征融合
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── requirements.txt     # 依赖列表
└── README.md            # 项目文档（本文档）
```

## 核心组件说明

### 1. 数据处理模块 (data_processor.py)

负责加载和预处理训练数据，支持JSONL格式和Hugging Face数据集。预处理功能包括：
- 提取对话历史、记忆查询和答案
- 使用分词器处理输入文本
- 生成模型训练所需的输入特征和标签
- 保留对话历史信息用于记忆融合

### 2. Hippo模型 (hippo_model.py)

专门设计的记忆编码模型，用于处理对话历史信息：
- 将对话历史编码为向量表示
- 捕获对话中的关键信息和上下文
- 生成可与大语言模型融合的特征表示

### 3. 模型定制 (model_customization.py)

实现ModifiedQwen类，集成基础大语言模型和记忆融合功能：
- 继承自预训练Qwen模型
- 在forward方法中集成HippoModel
- 实现记忆特征与大模型输出的融合

### 4. 训练模块 (train.py)

自定义训练器实现：
- 实现CustomTrainer类，重写compute_loss方法
- 支持对话历史信息的传递和处理
- 配置训练参数和优化策略

## 数据格式

训练数据采用JSONL格式，每条记录包含以下字段：

```json
{
  "dialog_history": ["用户问题1", "助手回答1", "用户问题2", "助手回答2"...],
  "memory_query": "关于历史对话内容的查询",
  "memory_answer": "基于历史对话的正确回答"
}
```

## 使用场景

HIPPO系统特别适用于以下场景：
- 需要长对话记忆的客服机器人
- 个性化对话助手
- 需要保持对话连贯性的交互式应用
- 教育和辅导场景中的问答系统

## 性能优化建议

1. 根据硬件资源调整batch_size和max_length参数
2. 对于长对话历史，考虑使用历史压缩或摘要技术
3. 可根据具体任务微调HippoModel的参数
4. 考虑使用混合精度训练加速训练过程

## 注意事项

1. 系统依赖于预训练的Qwen模型，请确保有正确的模型访问权限
2. 对话历史长度可能影响系统性能，建议合理控制
3. 训练数据质量对模型性能有显著影响，请确保数据质量

## 许可证

[MIT License](LICENSE)

## 联系信息

如有问题或建议，请联系项目维护者。