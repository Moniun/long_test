import torch
import os
import json
import gc
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM
)
from torch.cuda.amp import autocast
from torch import nn
from data_processor import load_train_val_data
import argparse
import nltk
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_transformer_util
from sentence_transformers.util import cos_sim as calculate_cosine_similarity
from model_customization import ModifiedQwen
from lora_model_customization import HippoLoRAQwen


# 确保NLTK数据可用
try:
    nltk.data.path.append("./nltk_data")
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 延迟初始化Sentence-BERT模型，避免在导入时占用内存
sbert_model = None

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return sbert_model


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型和数据路径
    parser.add_argument("--model_name_or_path", type=str, default="Qwen")
    parser.add_argument("--data_path", type=str, default="data/data_train.jsonl", help="训练数据路径")
    parser.add_argument("--test_data_path", type=str, default="data/data_val.jsonl", help="验证数据路径")
    parser.add_argument("--base_output_dir", type=str, default="./models", help="基础输出目录")
    parser.add_argument("--cache_dir", type=str, default="qwen3-8b-custom-module-training", help="模型缓存目录")
    parser.add_argument("--train_processed_path", type=str, default="./processed_data/train_dataset", help="训练数据预处理后保存和加载的路径")
    parser.add_argument("--val_processed_path", type=str, default="./processed_data/val_dataset", help="验证数据预处理后保存和加载的路径")
    
    # Hippo和LoRA相关参数
    parser.add_argument("--fusion_layers", type=str, default="2", 
                       help="指定要应用Hippo/LoRA的层列表，用逗号分隔，如'10,20,30'。在lora和full方法中都会被使用")
    parser.add_argument("--last_n_tokens", type=int, default=0, 
                       help="指定Hippo模型在训练时只关注input_ids的最后N个token，0表示处理所有token")
    
    # LoRA微调参数
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA的rank参数")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA的alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA的dropout参数")
    parser.add_argument("--attn_lora_config", type=str, default=None, 
                       help="注意力层LoRA配置，JSON格式，如'{\"rank\": 16, \"alpha\": 32}'")
    parser.add_argument("--ffn_lora_config", type=str, default=None, 
                       help="FFN层LoRA配置，JSON格式，如'{\"rank\": 8, \"alpha\": 64}'")
    
    # 训练控制参数
    parser.add_argument("--finetuning_method", type=str, default="lora", choices=["lora", "full"], 
                       help="微调方法选择：lora表示使用lora_model_customization.py进行LoRA微调，"
                            "full表示使用model_customization.py进行全参数微调")
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="评估时的batch size（减小以节省内存）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_accumulation_steps", type=int, default=8, help="评估时的梯度累积步数（增加以模拟更大batch）")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减参数")
    parser.add_argument("--transformer_learning_rate", type=float, default=1e-6, help="Transformer层的学习率")
    parser.add_argument("--hippo_learning_rate", type=float, default=1e-6, help="HippoModel的学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="学习率预热比例")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大训练步数，-1表示由epoch决定")
    
    # 混合精度和内存优化
    parser.add_argument("--fp16", action="store_true", default=True, help="是否启用混合精度训练(fp16)")
    parser.add_argument("--bf16", action="store_true", default=False, help="是否启用混合精度训练(bf16)")
    parser.add_argument("--max_split_size_mb", type=int, default=256, help="CUDA内存分割大小")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="是否启用梯度检查点")
    
    # 评估和日志
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "epoch", "steps"], help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=20, help="每多少步进行一次评估")
    parser.add_argument("--test_sample_ratio", type=float, default=1.0, help="测试集采样比例，0.0-1.0")
    parser.add_argument("--logging_steps", type=int, default=1, help="日志打印步数")
    parser.add_argument("--save_steps", type=int, default=32, help="模型保存步数")
    parser.add_argument("--save_total_limit", type=int, default=2, help="最大保存检查点数量")
    
    # 训练控制
    parser.add_argument("--load_best_model", action="store_true", help="是否在训练结束时加载最佳模型")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练的路径")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_meteor", help="用于选择最佳模型的指标")
    
    # 内存优化参数
    parser.add_argument("--eval_sample_limit", type=int, default=200, help="评估时最大样本数，用于内存优化")
    parser.add_argument("--metric_batch_size", type=int, default=4, help="评估时指标计算的小批量大小，用于内存优化")

    return parser.parse_args()

# 优化的CustomTrainer类，专门处理内存问题
class OptimizedTrainer(Trainer):
    def __init__(self, *args, cmd_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmd_args = cmd_args
        self.first_epoch_completed = False
        self.transformer_layers_unfrozen = False
        self.current_epoch = 0
        self.custom_tokenizer = AutoTokenizer.from_pretrained(cmd_args.model_name_or_path)
        # 延迟加载Sentence-BERT模型
        self._sbert_model = get_sbert_model()

        self.compute_metrics = self._compute_metrics
        
    @property
    def sbert_model(self):
        if self._sbert_model is None:
            self._sbert_model = get_sbert_model()
        return self._sbert_model
    
    def _compute_metrics(self, eval_pred, compute_result=False):
        """内存优化的评估指标计算（支持批次累积）"""
        predictions, labels = eval_pred

        # ==================== 初始化累积变量（首次调用时） ====================
        if not hasattr(self, "accumulated_metrics"):
            self.accumulated_metrics = {
                # F1相关累积值
                "true_positives": 0,
                "predicted_positives": 0,
                "actual_positives": 0,
                # METEOR相关累积值（用总和+计数代替列表，节省内存）
                "meteor_sum": 0.0,
                "meteor_count": 0,
                # SBERT相关累积值
                "sbert_sum": 0.0,
                "sbert_count": 0,
                # 总样本数
                "total_samples": 0
            }

        # ==================== 中间批次逻辑（累积结果，不返回最终指标） ====================
        if not compute_result:
            # 1. 限制单批次样本量（避免超过总限制）
            max_total_samples = self.cmd_args.eval_sample_limit
            remaining_samples = max_total_samples - self.accumulated_metrics["total_samples"]
            if remaining_samples <= 0:
                return  # 已达样本上限，停止累积
            
            current_batch_size = len(predictions)
            actual_use_samples = min(current_batch_size, remaining_samples)
            if actual_use_samples < current_batch_size:
                print(f"批次样本截断：{current_batch_size} -> {actual_use_samples}（总样本已达上限）")
                predictions = predictions[:actual_use_samples]
                labels = labels[:actual_use_samples]
            
            # 2. 计算当前批次的Token级指标（F1相关）
            predictions = torch.argmax(torch.tensor(predictions), dim=-1)
            mask = labels != -100  # 过滤无效标签（-100）
            
            batch_true_positives = ((predictions == labels) & mask).sum().item()
            batch_predicted_positives = mask.sum().item()  # 预测的有效位置数
            batch_actual_positives = mask.sum().item()     # 实际的有效位置数（与预测位置一致，因mask相同）
            
            # 3. 累积Token级指标
            self.accumulated_metrics["true_positives"] += batch_true_positives
            self.accumulated_metrics["predicted_positives"] += batch_predicted_positives
            self.accumulated_metrics["actual_positives"] += batch_actual_positives
            
            # 4. 处理当前批次的语义级指标（METEOR和SBERT）
            batch_size = self.cmd_args.metric_batch_size  # 内部小批量处理，减少内存压力
            total_batch_in_chunk = (len(predictions) + batch_size - 1) // batch_size
            
            for chunk_idx in range(total_batch_in_chunk):
                start = chunk_idx * batch_size
                end = min(start + batch_size, len(predictions))
                chunk_preds = predictions[start:end]
                chunk_labels = labels[start:end]
                chunk_masks = mask[start:end]
                
                for i in range(len(chunk_preds)):
                    pred = chunk_preds[i]
                    label = chunk_labels[i]
                    msk = chunk_masks[i]
                    
                    # 过滤无效位置，解码文本
                    valid_pred_ids = pred[msk].tolist()
                    valid_label_ids = label[msk].tolist()
                    
                    try:
                        pred_text = self.custom_tokenizer.decode(valid_pred_ids, skip_special_tokens=True).strip()
                        label_text = self.custom_tokenizer.decode(valid_label_ids, skip_special_tokens=True).strip()
                        
                        if not (pred_text and label_text):
                            continue  # 跳过空文本
                        
                        # 计算METEOR并累积
                        pred_tokens = pred_text.split()
                        label_tokens = label_text.split()
                        if pred_tokens and label_tokens:
                            try:
                                meteor = meteor_score([label_tokens], pred_tokens)
                                self.accumulated_metrics["meteor_sum"] += meteor
                                self.accumulated_metrics["meteor_count"] += 1
                            except Exception:
                                continue
                        
                        # 计算SBERT相似度并累积
                        try:
                            pred_embedding = self._sbert_model.encode(pred_text, convert_to_tensor=True)
                            label_embedding = self._sbert_model.encode(label_text, convert_to_tensor=True)
                            similarity = calculate_cosine_similarity(pred_embedding, label_embedding).item()
                            self.accumulated_metrics["sbert_sum"] += similarity
                            self.accumulated_metrics["sbert_count"] += 1
                            
                            # 及时释放显存
                            del pred_embedding, label_embedding
                            torch.cuda.empty_cache()
                        except Exception:
                            continue
                        
                    except Exception:
                        continue
                
                # 每处理10个小批次清理一次内存
                if chunk_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # 5. 累积总样本数
            self.accumulated_metrics["total_samples"] += actual_use_samples
            return  # 中间批次不返回指标


        # ==================== 最后批次逻辑（汇总计算最终指标） ====================
        else:
            # 1. 从累积结果计算全局F1
            tp = self.accumulated_metrics["true_positives"]
            pp = self.accumulated_metrics["predicted_positives"]
            ap = self.accumulated_metrics["actual_positives"]
            
            precision = tp / pp if pp > 0 else 0.0
            recall = tp / ap if ap > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 2. 计算全局METEOR和SBERT平均
            avg_meteor = (self.accumulated_metrics["meteor_sum"] / self.accumulated_metrics["meteor_count"]
                            if self.accumulated_metrics["meteor_count"] > 0 else 0.0)
            
            avg_sbert = (self.accumulated_metrics["sbert_sum"] / self.accumulated_metrics["sbert_count"]
                            if self.accumulated_metrics["sbert_count"] > 0 else 0.0)
            
            # 3. 输出评估结果
            print(f"\n===== 评估结果 (Epoch {self.current_epoch}) =====")
            print(f"- F1分数 (Token级): {f1_score:.4f}")
            print(f"- METEOR分数 (语义级): {avg_meteor:.4f}")
            print(f"- Sentence-BERT相似度 (语义级): {avg_sbert:.4f}")
            print(f"- 评估样本数: {self.accumulated_metrics['total_samples']}")
            print("====================\n")
            
            # 4. 重置累积变量（避免影响下一次评估）
            del self.accumulated_metrics
            
            # 5. 返回最终指标
            return {
                "f1": f1_score,
                "meteor": avg_meteor,
                "sbert_similarity": avg_sbert
            }
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """内存优化的评估流程"""
        print(f"\n开始内存优化的评估 (Epoch {self.current_epoch})...")
        
        # 评估前清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # 限制评估数据集大小
        if hasattr(eval_dataset, 'num_rows') and eval_dataset.num_rows > self.cmd_args.eval_sample_limit:
            print(f"限制评估数据集大小: {eval_dataset.num_rows} -> {self.cmd_args.eval_sample_limit}")
            eval_dataset = eval_dataset.select(range(self.cmd_args.eval_sample_limit))
        
        with torch.no_grad():
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 评估后再次清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        return metrics
    
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        print(f"\n===== 开始训练第 {self.current_epoch} 个epoch =====", flush=True)
        
        # 内存清理
        torch.cuda.empty_cache()
        gc.collect()
    
    def on_step_end(self, args, state, control, **kwargs):
        # 更频繁的内存清理
        if state.global_step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        return control


def main():
    args = parse_args()
    
    print("=== 内存优化配置 ===")
    print(f"评估样本限制: {args.eval_sample_limit}")
    print(f"评估batch size: {args.per_device_eval_batch_size}")
    print(f"评估梯度累积步数: {args.eval_accumulation_steps}")
    print(f"指标计算批量大小: {args.metric_batch_size}")
    print("====================\n")
    
    # 解析fusion_layers参数
    try:
        fusion_layers_list = [int(x.strip()) for x in args.fusion_layers.split(',') if x.strip()]
        print(f"Fusion layers: {fusion_layers_list}")
    except ValueError as e:
        print(f"fusion_layers参数解析错误: {e}")
        print("使用默认fusion_layers: [10, 20, 30]")
        fusion_layers_list = [10, 20, 30]
    
    # 根据微调方法设置输出目录
    model_type_dir = {
        "lora": "lora_finetuning",
        "full": "full_parameter_finetuning"
    }
    
    # 确保基础输出目录和对应的子目录存在
    os.makedirs(args.base_output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_output_dir, "hippo_model"), exist_ok=True)
    os.makedirs(os.path.join(args.base_output_dir, model_type_dir[args.finetuning_method]), exist_ok=True)
    
    # 设置最终的输出目录
    args.output_dir = os.path.join(args.base_output_dir, model_type_dir[args.finetuning_method])
    
    print(f"基础输出目录: {args.base_output_dir}")
    print(f"微调方法: {args.finetuning_method}")
    print(f"模型保存目录: {args.output_dir}")
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"expandable_segments:True,max_split_size_mb:{args.max_split_size_mb}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    train_dataset, test_dataset = load_train_val_data(
        train_path=args.data_path,
        val_path=args.test_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_processed_path=args.train_processed_path,
        val_processed_path=args.val_processed_path
    )
    
    # 采样测试集
    if args.test_sample_ratio < 1.0:
        import random
        random.seed(42)
        sample_size = int(len(test_dataset) * args.test_sample_ratio)
        test_dataset = random.sample(test_dataset, sample_size)
        print(f"已对测试集进行采样，采样比例: {args.test_sample_ratio}, 采样后大小: {len(test_dataset)}")
    
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    
    # 加载模型
    if args.resume_from_checkpoint is not None:
        print(f"从检查点恢复训练: {args.resume_from_checkpoint}")
        if args.finetuning_method == "lora":
            model = HippoLoRAQwen.from_pretrained(args.resume_from_checkpoint, fusion_layers=fusion_layers_list)
        else:
            model = ModifiedQwen.from_pretrained(args.resume_from_checkpoint, fusion_layers=fusion_layers_list)
    else:
        print("从零开始训练新模型")
        if args.finetuning_method == "lora":
            print("使用LoRA微调方法")
            
            # 解析JSON格式的LoRA配置
            attn_config = None
            ffn_config = None
            
            try:
                if args.attn_lora_config:
                    attn_config = json.loads(args.attn_lora_config)
                    print(f"注意力层LoRA配置: {attn_config}")
            except json.JSONDecodeError as e:
                raise ValueError(f"attn_lora_config JSON格式错误: {e}")
                
            try:
                if args.ffn_lora_config:
                    ffn_config = json.loads(args.ffn_lora_config)
                    print(f"FFN层LoRA配置: {ffn_config}")
            except json.JSONDecodeError as e:
                raise ValueError(f"ffn_lora_config JSON格式错误: {e}")
            
            model = HippoLoRAQwen(
                base_model_name_or_path=args.model_name_or_path, 
                fusion_layers=fusion_layers_list,
                seq_len=args.max_length,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                cache_dir=args.cache_dir,
                last_n_tokens=args.last_n_tokens,
                attn_lora_config=attn_config,
                ffn_lora_config=ffn_config
            )

        else:
            print("使用全参数微调方法")
            model = ModifiedQwen(
                base_model_name_or_path=args.model_name_or_path,
                fusion_layers=fusion_layers_list,
                cache_dir=args.cache_dir,
                last_n_tokens=args.last_n_tokens
            )
    
    # 自定义数据收集器
    class CustomDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            filtered_features = []
            for feature in features:
                filtered_feature = {
                    'input_ids': feature['input_ids'],
                    'attention_mask': feature['attention_mask'],
                    'labels': feature['labels'],
                    'dialog_histories': feature.get('dialog_histories', [])
                }
                filtered_features.append(filtered_feature)
            return super().__call__(filtered_features)
    
    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
    
    # 计算总步数和预热步数
    total_steps = args.max_steps if args.max_steps > 0 else \
        (len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs)
    warmup_steps = int(args.warmup_ratio * total_steps) if args.warmup_ratio > 0 else 0
    
    # 训练参数配置（针对内存优化）
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.hippo_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_steps=args.max_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16_full_eval=True,  # 评估时使用fp16
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        eval_accumulation_steps=args.eval_accumulation_steps,  # 评估时的梯度累积
        load_best_model_at_end=args.load_best_model,
        metric_for_best_model=args.metric_for_best_model if args.metric_for_best_model.startswith("eval_") else f"eval_{args.metric_for_best_model}",
        greater_is_better=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,  # 禁用pin_memory以节省内存
        batch_eval_metrics=True  # 启用批量评估指标
    )
    
    # 初始化训练环境
    print("开始训练")
    trainer = OptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        # compute_metrics=OptimizedTrainer.compute_metrics,
        cmd_args=args
    )
    
    # 打印训练配置信息
    print(f"训练设置: 学习率={args.hippo_learning_rate}, 预热步数={warmup_steps}")
    print(f"总训练步数: {total_steps}")
    print(f"内存优化: eval_accumulation_steps={args.eval_accumulation_steps}")
    
    # 开始训练
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # 保存模型
        torch.cuda.empty_cache()
        try:
            model_to_save = trainer.model
            if hasattr(model_to_save, 'module'):
                model_to_save = model_to_save.module
            
            model_to_save.to('cpu')
            torch.cuda.empty_cache()
            
            if args.finetuning_method == "lora":
                print("保存LoRA微调模型...")
                model_to_save.save_pretrained(
                    save_directory=args.base_output_dir, 
                    model_type="lora",
                    save_hippo_components=True
                )
                model_to_save.save_pretrained(
                    save_directory=args.base_output_dir, 
                    model_type="hippo",
                    save_hippo_components=True
                )
            else:
                print("保存全参数微调模型...")
                model_to_save.save_pretrained(
                    save_directory=args.base_output_dir, 
                    model_type="full"
                )
                model_to_save.save_pretrained(
                    save_directory=args.base_output_dir, 
                    model_type="hippo",
                    save_hippo_components=True
                )
            
            tokenizer.save_pretrained(args.output_dir)
            
            print(f"\n模型保存完成！")
            print(f"基础输出目录: {args.base_output_dir}")
            print(f"- Hippo组件保存在: {os.path.join(args.base_output_dir, 'hippo_model')}")
            if args.finetuning_method == "lora":
                print(f"- LoRA微调模型保存在: {os.path.join(args.base_output_dir, 'lora_finetuning')}")
            else:
                print(f"- 全参数微调模型保存在: {os.path.join(args.base_output_dir, 'full_parameter_finetuning')}")
                
        except Exception as e:
            print(f"保存模型时出错: {e}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n内存不足错误！已提供内存优化版本，尝试以下解决方案：")
            print("1. 使用train_optimized.py而不是train.py")
            print("2. 进一步减小per_device_eval_batch_size")
            print("3. 减小eval_sample_limit参数")
            print("4. 减小Sentence-BERT评估的批次大小")
            print("5. 增加eval_accumulation_steps参数")

        raise e
    finally:
        # 清理工作
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()