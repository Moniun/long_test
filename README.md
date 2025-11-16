# HIPPO: 对话历史记忆融合系统

HIPPO (History-Integrated Processing for Personalized Outputs) 是一个结合对话历史记忆功能的大语言模型增强系统。该系统通过引入专门的记忆模型（HippoModel），使大语言模型能够更好地理解和利用对话历史信息，提供更加连贯和个性化的回答。

## 项目概述

HIPPO系统的核心创新点在于：
- 将对话历史信息通过专门的记忆模型处理，生成记忆表示
- 将记忆表示与大语言模型的输出特征进行融合，实现记忆增强
- 采用非侵入式设计，不需要修改大语言模型的底层结构
- 支持阶段性训练策略，先训练记忆模型，再进行联合优化
- 实现差异化学习率，加速关键组件的收敛
- **最新特性：基于矩阵的轻量化Hippo实现，参数量减少99.9%**

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

## 版本更新：矩阵优化版本 🚀

我们推出了基于矩阵计算的Hippo模型重构版本，实现了显著的性能优化：

### 核心改进

| 指标 | 原始Hippo模型 | 矩阵优化版本 | 改善 |
|------|---------------|--------------|------|
| 参数数量 | ~36M | ~1-2K | **99.9%减少** |
| FP32内存 | ~138MB | ~0.01MB | **99.9%减少** |
| FP16内存 | ~69MB | ~0.005MB | **99.9%减少** |
| 计算复杂度 | 复杂选择性扫描 | 简单矩阵乘法 | **显著简化** |

### 主要特性

- ✅ **参数效率极高**：删除36M参数，仅保留1-2K核心参数
- ✅ **内存占用极低**：适合资源受限环境部署
- ✅ **计算直观简单**：矩阵乘法清晰易懂，便于调试
- ✅ **接口完全兼容**：无缝替换原始Hippo模型
- ✅ **实现简单**：无复杂门控机制，易于维护

### 技术细节

```python
# 矩阵版本的核心计算过程
# h = A*h + B*x  (状态更新)
# y = C*h + D*x  (输出计算)

# 参数矩阵：
# B: (hidden_dim, seq_len) - 输入控制矩阵
# H: (hidden_dim, input_dim) - 状态转换矩阵  
# C: (seq_len, hidden_dim) - 输出控制矩阵
# D: (seq_len, seq_len) - 直接投影矩阵
```

### 相关文件

- `hippo_model.py` - 包含原始Hippo模型和矩阵版本
- `matrix_hippo_model.py` - 独立矩阵实现
- `matrix_hippo_model_summary.md` - 详细技术说明
- `compare_models.py` - 性能对比工具
- `linear_vs_matrix_analysis.md` - 参数优化分析

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

### 数据准备

系统支持JSONL格式的训练数据。按照以下步骤准备数据：

1. 将训练数据和验证数据放在项目根目录的`data`文件夹中
2. 确保训练数据文件名为`data_train.jsonl`
3. 确保验证数据文件名为`data_val.jsonl`

数据格式要求：
```json
{
  "dialog_history": ["用户问题1", "助手回答1", "用户问题2", "助手回答2"...],
  "memory_query": "关于历史对话内容的查询",
  "memory_answer": "基于历史对话的正确回答"
}
```

### 模型训练

使用以下命令启动模型训练（使用默认参数时只需简单运行）：

```bash
python train.py
```

训练将自动加载`data/data_train.jsonl`作为训练集和`data/data_val.jsonl`作为验证集，并将模型保存到`./models`目录。

可选参数配置：
- `--model_name_or_path`: 预训练模型路径，默认为"Qwen/Qwen3-8B-Chat"
- `--data_path`: 训练数据路径，默认为"data/data_train.jsonl"
- `--test_data_path`: 验证数据路径，默认为"data/data_val.jsonl"
- `--base_output_dir`: 基础模型保存路径，默认为"./models"
- `--finetuning_method`: 微调方法，支持"lora"和"full"，默认为"full"
- `--num_train_epochs`: 训练轮数，默认为3
- `--per_device_train_batch_size`: 每设备训练批次大小，默认为8
- `--gradient_accumulation_steps`: 梯度累积步数，默认为2
- `--learning_rate`: 学习率，默认为2e-4
- `--max_length`: 最大序列长度，默认为None（自动检测）
- `--evaluation_strategy`: 评估策略，默认为"steps"
- `--eval_steps`: 每多少步进行一次评估，默认为500
- `--load_best_model`: 是否在训练结束时加载最佳模型

训练过程中的关键特性：
- 第一个epoch只训练Hippo模型和门控机制
- 后续epoch自动解冻Transformer层进行联合训练
- 差异化学习率：Hippo模型使用5e-5，Transformer层使用1e-5
- 自动保存最佳模型检查点
- **分类保存机制**：根据微调方法将模型保存到不同目录

### 推理使用

训练完成后，可以使用`inference.py`进行推理。系统现在使用隐藏状态机制维护对话上下文，不再需要显式提供对话历史。

#### 单次执行模式

使用`--prompt`参数直接提供输入内容：

```bash
python inference.py --model_path ./models --prompt "你好，请问有什么可以帮助你的？"
```

#### 交互式模式

直接运行`inference.py`即可进入交互式对话模式：

```bash
python inference.py --model_path ./models
```

在交互式模式中，系统提供以下命令：
- `exit` 或 `quit`: 退出程序
- `reset_hidden`: 重置HIPPO模型的隐藏状态

HIPPO模型会自动通过隐藏状态维护对话上下文，无需显式提供历史记录。这是系统的核心特性，允许模型在长对话中保持记忆。

## 模型分类保存

为了更好地组织和管理不同微调方式的模型，我们实现了分类保存机制。根据选择的微调方法，模型会被保存到不同的目录结构中。

### 目录结构

```
models/
├── hippo_model/                 # Hippo组件通用目录
│   ├── hippo_components.bin    # Hippo模型和门控机制参数
│   ├── config.json             # Hippo组件配置信息
│   └── hippo_lora_components.pt # LoRA版本的Hippo组件
├── lora_finetuning/            # LoRA微调模型保存目录
│   ├── base_model/             # 基础模型权重
│   ├── hippo_lora_components.pt # Hippo组件+LoRA参数
│   ├── config.json             # 模型配置
│   └── tokenizer/              # 分词器
└── full_parameter_finetuning/  # 全参数微调模型保存目录
    ├── base_model/             # 基础模型权重
    ├── custom_modules.bin      # Hippo自定义模块
    ├── config.json             # 模型配置
    └── tokenizer/              # 分词器
```

### 训练命令示例

#### LoRA微调训练
```bash
# LoRA微调 - 模型会保存到lora_finetuning和hippo_model目录
python train.py \
    --finetuning_method lora \
    --model_name_or_path "Qwen/Qwen2-1.5B-Instruct" \
    --base_output_dir "./models"
```

#### 全参数微调训练
```bash
# 全参数微调 - 模型会保存到full_parameter_finetuning和hippo_model目录
python train.py \
    --finetuning_method full \
    --model_name_or_path "Qwen/Qwen2-1.5B-Instruct" \
    --base_output_dir "./models"
```

### 推理命令示例

#### LoRA模型推理
```bash
# 使用lora_finetuning目录中的模型
python inference.py \
    --model_type lora \
    --model_path "./models/lora_finetuning"
```

#### 全参数模型推理
```bash
# 使用full_parameter_finetuning目录中的模型
python inference.py \
    --model_type full \
    --model_path "./models/full_parameter_finetuning"
```

### 微调方法对比

| 维度 | LoRA微调 | 全参数微调 |
|------|----------|------------|
| 可训练参数 | ~5-10% | 100% |
| 显存占用 | 低 | 高 |
| 训练速度 | 快 | 慢 |
| 存储空间 | 小 | 大 |
| 性能表现 | 良好 | 最佳 |
| hippo_model/ | ✓ | ✓ |
| 专有目录 | lora_finetuning/ | full_parameter_finetuning/ |

### 模型加载说明

#### HippoLoRAQwen模型（LoRA微调）
```python
from lora_model_customization import HippoLoRAQwen

# 加载LoRA微调模型
model = HippoLoRAQwen.from_pretrained("./models/lora_finetuning")

# 单独加载Hippo组件
model = HippoLoRAQwen.from_pretrained("./models/hippo_model")
```

#### ModifiedQwen模型（全参数微调）
```python
from model_customization import ModifiedQwen

# 加载全参数微调模型
model = ModifiedQwen.from_pretrained("./models/full_parameter_finetuning")

# 单独加载Hippo组件
model = ModifiedQwen.from_pretrained("./models/hippo_model")
```

### 最佳实践

1. **选择LoRA微调**: 当资源有限或需要快速实验时
2. **选择全参数微调**: 当追求最佳性能且有充足计算资源时
3. **Hippo组件共享**: hippo_model/目录中的组件可以被不同微调方式共享
4. **版本管理**: 建议在不同目录下维护不同版本的模型，便于对比和回滚
5. **矩阵版本优先**: 新项目推荐使用矩阵版本的Hippo模型，性能更优

### 注意事项

- 训练完成后会同时保存完整模型和Hippo组件
- Hippo组件可以独立加载和部署
- 推理时需要根据`--model_type`参数选择正确的模型类型
- 不同微调方式的模型不可混用，需要使用对应的推理代码
- **矩阵版本Hippo模型**与原始版本接口完全兼容，可直接替换

## 文件结构

```
├── data/                    # 数据文件夹
│   ├── data_train.jsonl    # 训练数据集
│   └── data_val.jsonl      # 验证数据集
├── models/                  # 模型保存目录（分类保存）
│   ├── hippo_model/        # Hippo组件通用目录
│   ├── lora_finetuning/    # LoRA微调模型
│   └── full_parameter_finetuning/ # 全参数微调模型
├── data_processor.py        # 数据加载和预处理模块
├── generate_data.py         # 数据生成工具
├── hippo_model.py           # 记忆编码模型实现（包含矩阵版本）
├── matrix_hippo_model.py    # 独立矩阵版本实现
├── matrix_hippo_model_summary.md # 矩阵版本技术文档
├── compare_models.py        # 原始vs矩阵版本性能对比
├── linear_vs_matrix_analysis.md # 参数优化分析报告
├── model_customization.py   # 大模型定制和特征融合（全参数微调）
├── lora_model_customization.py # 大模型定制（LoRA微调）
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
├── requirements.txt         # 依赖列表
└── README.md                # 项目文档
```

## 核心功能说明

### 1. 数据处理功能 (data_processor.py)

- 支持同时加载和预处理训练集和验证集
- 兼容JSONL格式和Hugging Face数据集
- 优化的标签生成策略，提高训练效率
- 保留对话历史信息用于记忆模型处理
- 详细的日志输出，方便监控数据加载状态

### 2. 记忆编码模型 (hippo_model.py)

#### 原始Hippo模型
- 将对话历史编码为向量表示
- 捕获对话中的关键信息和上下文关联
- 生成可与大语言模型融合的特征表示
- 轻量级设计，平衡记忆能力和计算效率

#### 矩阵优化版本 (MatrixHippoModel)
- **参数大幅减少**：从36M减少到1-2K，减少99.9%
- **内存占用极低**：适合移动设备和边缘计算
- **计算直观**：简单的矩阵乘法，易于理解和调试
- **接口兼容**：与原始版本完全兼容，可直接替换
- **计算过程**：h = A*h + B*x 和 y = C*h + D*x

### 3. 模型集成与融合 (model_customization.py)

- 集成基础大语言模型和记忆融合功能
- 实现记忆特征与大模型输出的门控融合机制
- 智能的参数冻结/解冻控制，支持训练阶段管理
- 显存优化，降低推理时内存占用

### 4. 高级训练系统 (train.py)

- 阶段性训练策略：先训练记忆模型，再联合优化
- 差异化学习率自动应用，无需手动配置
- 自动层解冻机制，优化训练过程
- 动态评估和模型保存，跟踪训练进度

### 5. 记忆增强推理 (inference.py)

- 基于隐藏状态的上下文维护，无需显式提供对话历史
- 交互式对话模式，支持长对话记忆保持
- 自定义生成函数，精确控制隐藏状态更新
- 隐藏状态重置功能，灵活管理对话上下文

## 训练流程详解

1. **数据加载阶段**：
   - 自动加载`data/data_train.jsonl`和`data/data_val.jsonl`
   - 预处理对话历史、记忆查询和答案
   - 输出数据集大小统计信息

2. **模型初始化阶段**：
   - 加载基础Qwen模型
   - 初始化Hippo模型和门控机制（支持矩阵版本）
   - 冻结Transformer层，只保留Hippo模型参数可训练

3. **第一阶段训练**：
   - 只训练Hippo模型和门控机制
   - 让记忆模型先适应大模型的特征
   - 定期在验证集上评估性能

4. **自动阶段转换**：
   - 第一个epoch完成后，自动解冻Transformer层
   - 更新优化器参数组，应用差异化学习率
   - 继续训练所有可训练组件

5. **模型保存**：
   - 训练完成后将模型保存到`./models`目录
   - 可选保存最佳模型检查点

## 性能优化特性

1. **阶段性训练**：先训练记忆模型，再联合优化，稳定训练过程
2. **差异化学习率**：Hippo模型(5e-5)和Transformer层(1e-5)使用不同学习率
3. **显存优化**：混合精度训练(fp16)、梯度累积、智能梯度计算控制
4. **动态评估**：训练过程中定期在验证集上评估，监控模型性能
5. **矩阵版本优化**：99.9%参数减少，显著降低计算和存储成本

## Hippo模型版本对比

| 特性 | 原始版本 | 矩阵版本 | 推荐场景 |
|------|----------|----------|----------|
| 参数量 | ~36M | ~1-2K | 资源受限环境优先 |
| 内存占用 | ~138MB | ~0.01MB | 移动/边缘部署 |
| 计算复杂度 | 高 | 低 | 快速推理需求 |
| 表达能力 | 强 | 中等 | 复杂任务 |
| 维护难度 | 中等 | 简单 | 长期维护项目 |
| 兼容性 | 基础 | 完全兼容 | 无缝升级 |

## 常见问题与解决方案

1. **显存不足**：
   - 使用矩阵版本的Hippo模型（推荐）
   - 减小`per_device_train_batch_size`
   - 增加`gradient_accumulation_steps`
   - 使用`--max_length`限制序列长度

2. **训练不稳定**：
   - 矩阵版本训练更稳定
   - 减小学习率
   - 确保数据质量，检查数据格式正确性
   - 使用`--load_best_model`参数保存最佳模型

3. **记忆效果不佳**：
   - 增加对话历史长度
   - 提高训练轮数
   - 优化训练数据，增加更多样化的记忆查询样本

4. **版本选择**：
   - **新项目**：推荐矩阵版本，性能和效率更优
   - **现有项目**：可直接替换为矩阵版本，接口兼容
   - **性能要求高**：可考虑保留原始版本

## 许可证

[MIT License](LICENSE)

## 联系信息

如有问题或建议，请联系项目维护者。