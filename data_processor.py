from datasets import Dataset, load_dataset
import json
import torch

def load_and_preprocess_data(data_path, tokenizer, max_length=None, split=None, processed_data_path=None, max_history_rounds=12):
    """
    加载并预处理数据，支持dialog_history、memory_query、memory_answer格式
    
    Args:
        data_path: 数据文件路径（JSONL格式）或数据集名称
        tokenizer: 分词器
        max_length: 最大序列长度（默认为None，会自动从tokenizer获取模型最大长度）
        split: 数据集分割名称（用于Hugging Face数据集，如'train'、'validation'、'test'）
              对于JSONL文件，此参数无效
        processed_data_path: 预处理数据保存和加载的路径
                            如果该路径存在已处理好的数据，则直接使用；
                            如果不存在，则处理数据并保存到该路径
        max_history_rounds: 最大对话轮数（默认为12）
    """
    # 如果未指定max_length，尝试从tokenizer或其配置中获取模型最大长度
    if max_length is None:
        # 尝试获取模型的最大上下文长度（不同模型有不同的配置属性）
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length > 0:
            max_length = tokenizer.model_max_length
        elif hasattr(tokenizer, 'max_model_input_sizes') and tokenizer.max_model_input_sizes:
            # 对于某些旧版本的tokenizer
            first_model = list(tokenizer.max_model_input_sizes.keys())[0]
            max_length = tokenizer.max_model_input_sizes[first_model]
        else:
            # 如果无法自动获取，使用一个合理的默认值
            max_length = 4096
        print(f"自动检测到模型最大长度: {max_length}")
    # 加载数据
    if data_path.endswith(".jsonl"):
        # 直接读取JSONL文件，支持训练集和验证集文件
        print(f"正在加载JSONL文件: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        print(f"成功加载 {len(data)} 条数据")
    else:
        # 支持Hugging Face数据集
        print(f"正在加载数据集: {data_path}")
        dataset = load_dataset(data_path)
        # 根据指定的split选择数据集分割，默认为'train'
        split_name = split if split is not None else 'train'
        if split_name not in dataset:
            # 如果指定的分割不存在，使用第一个可用的分割
            available_splits = list(dataset.keys())
            print(f"警告: 分割 '{split_name}' 不存在，使用第一个可用分割 '{available_splits[0]}'")
            split_name = available_splits[0]
        data = list(dataset[split_name])
        print(f"成功加载数据集分割 '{split_name}'，共 {len(data)} 条数据")

    # 转换为Dataset对象
    dataset = Dataset.from_list(data)
    
    # 预处理函数
    def preprocess_function(example):
        # 提取对话历史、记忆查询和答案
        dialog_history = example.get("dialog_history", [])
        memory_query = example.get("memory_query", "")
        memory_answer = example.get("memory_answer", "")
        
        # 使用传入的max_history_rounds参数（默认为12）
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 3. 初始化固定形状的二维张量（填充pad_token_id）
        encoded_dialog_history = torch.full((max_history_rounds, max_length), pad_token_id, dtype=torch.long)
        
        for i, history_item in enumerate(dialog_history[:max_history_rounds]):  # 限制不超过最大轮数
            if not history_item:  # 跳过空字符串
                continue
            # 编码单轮对话
            history_encoding = tokenizer(
                history_item,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            # 填入二维张量的第i行（覆盖pad填充）
            encoded_dialog_history[i] = history_encoding["input_ids"][0]

        # 处理对话文本
        messages = [
            # {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": memory_query},
            {"role": "assistant", "content": memory_answer}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            enable_thinking=False,
            add_generation_prompt=False
        ).squeeze(0)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
        labels = input_ids.clone()
        
        assistant_token_id = tokenizer.convert_tokens_to_ids(["assistant"])[0]

        try:
            assistant_start = (input_ids == assistant_token_id).nonzero()[0].item() + 1
        except IndexError:
            # 如果没找到，使用默认位置（通常不会发生）
            assistant_start = 0

        labels[:assistant_start+5] = -100

        return {
            "dialog_histories": encoded_dialog_history,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        
        # # 标准语言模型训练范式：完整序列输入，查询部分标签设为-100
        # # 构建完整输入序列：memory_query + eos_token + memory_answer + eos_token
        # full_input_text = f"{memory_query}{tokenizer.eos_token}{memory_answer}{tokenizer.eos_token}"
        
        # # 编码完整的输入文本
        # encoding = tokenizer(
        #     full_input_text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=max_length,
        #     return_tensors="pt"
        # )
        
        # # 输入和注意力掩码
        # input_ids = encoding["input_ids"][0]
        # attention_mask = encoding["attention_mask"][0]
        
        # # 标签：复制输入序列，将查询部分（包含eos_token）设为-100，答案部分参与损失计算
        # labels = input_ids.clone()
        
        # # 计算查询部分的长度（memory_query + eos_token）
        # query_with_sep = f"{memory_query}{tokenizer.eos_token}"
        # query_encoding = tokenizer(query_with_sep, add_special_tokens=False)
        # query_length = len(query_encoding["input_ids"])
        
        # # 将查询部分设为-100（不参与损失计算）
        # labels[:query_length] = -100
        
        # # 确保不超过序列长度
        # if labels.shape[0] > max_length:
        #     labels = labels[:max_length]
        
        # # 返回处理后的数据
        # return {
        #     "dialog_histories": encoded_dialog_history,
        #     # "memory_queries": memory_query,
        #     # "memory_answers": memory_answer,
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": labels
        # }
    
    # 首先尝试从已保存的预处理数据加载（如果指定了保存路径且存在）
    if processed_data_path:
        import os
        if (processed_data_path.endswith('.json') or processed_data_path.endswith('.jsonl')) and os.path.exists(processed_data_path):
            print(f"从已保存的JSON文件加载预处理数据: {processed_data_path}")
            from datasets import load_dataset
            tokenized_dataset = load_dataset('json', data_files=processed_data_path)
            # 将数据集字典转换为Dataset对象
            tokenized_dataset = tokenized_dataset['train']
            print(f"成功加载已预处理数据，共 {len(tokenized_dataset)} 条")
            return tokenized_dataset
        elif os.path.exists(processed_data_path) and os.path.isdir(processed_data_path):
            print(f"从已保存的Dataset目录加载预处理数据: {processed_data_path}")
            from datasets import load_from_disk
            tokenized_dataset = load_from_disk(processed_data_path)
            print(f"成功加载已预处理数据，共 {len(tokenized_dataset)} 条")
            return tokenized_dataset
        else:
            print(f"未找到已保存的预处理数据，将进行预处理并保存到 {processed_data_path}")
    
    # 应用预处理
    print(f"开始预处理数据...")
    # 注意：由于我们需要保留dialog_histories等非张量列，不使用remove_columns
    tokenized_dataset = dataset.map(preprocess_function)
    print(f"数据预处理完成")
    
    # 保存预处理数据（如果指定了保存路径）
    if processed_data_path:
        print(f"保存预处理数据到: {processed_data_path}")
        import os
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(processed_data_path)), exist_ok=True)
        
        # 根据文件扩展名选择保存格式
        if processed_data_path.endswith('.json') or processed_data_path.endswith('.jsonl'):
            tokenized_dataset.to_json(processed_data_path)
        else:
            # 默认保存为HuggingFace的Dataset格式（目录）
            tokenized_dataset.save_to_disk(processed_data_path)
        print(f"数据保存成功")
    
    return tokenized_dataset


def load_train_val_data(train_path, val_path, tokenizer, max_length=None, train_processed_path=None, val_processed_path=None, max_history_rounds=12):
    """
    加载并预处理训练集和验证集数据
    
    Args:
        train_path: 训练集文件路径（JSONL格式）
        val_path: 验证集文件路径（JSONL格式）
        tokenizer: 分词器
        max_length: 最大序列长度
        train_processed_path: 训练集预处理数据保存和加载的路径
                             如果该路径存在已处理好的数据，则直接使用；
                             如果不存在，则处理数据并保存到该路径
        val_processed_path: 验证集预处理数据保存和加载的路径
                            如果该路径存在已处理好的数据，则直接使用；
                            如果不存在，则处理数据并保存到该路径
        max_history_rounds: 最大对话轮数（默认为12）
                            
    Returns:
        tuple: (tokenized_train_dataset, tokenized_val_dataset)
    """
    # 加载并预处理训练集（支持从已保存的预处理数据加载）
    print(f"处理训练集: {train_path}")
    train_dataset = load_and_preprocess_data(train_path, tokenizer, max_length, processed_data_path=train_processed_path, max_history_rounds=max_history_rounds)
    
    # 加载并预处理验证集（支持从已保存的预处理数据加载）
    print(f"处理验证集: {val_path}")
    val_dataset = load_and_preprocess_data(val_path, tokenizer, max_length, processed_data_path=val_processed_path, max_history_rounds=max_history_rounds)
    
    return train_dataset, val_dataset