import torch
from model_customization import ModifiedQwen
import argparse
from transformers.modeling_outputs import CausalLMOutputWithPast

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--model_type", type=str, default="lora", choices=["lora", "full"], 
                       help="模型类型：lora表示使用lora_model_customization.py训练的模型，"
                            "full表示使用model_customization.py训练的模型")
    parser.add_argument("--prompt", type=str, default=None, help="提示词，如果不提供则进入交互式模式")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--check_params", action="store_true", help="验证模型参数完整性")
    return parser.parse_args()

def custom_generate(model, model_type, tokenizer, input_text, max_new_tokens=4096, temperature=0.7, reset_state=False):
    """自定义生成函数，直接使用forward而非generate，精确控制隐藏状态"""
    
    # 可选重置隐藏状态（仅full参数模型支持）
    if reset_state and model_type == "full":
        if hasattr(model, 'hippo_model') and hasattr(model.hippo_model, 'reset_h'):
            model.hippo_model.hidden_h = model.hippo_model.reset_h(batch_size=1)
    
    # 编码初始输入
    inputs = tokenizer(input_text, return_tensors="pt").to(model.base_model.device if hasattr(model, 'base_model') else 'cuda')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # 生成序列
    generated_tokens = []
    
    for step in range(max_new_tokens):
        # 直接调用forward获取logits
        with torch.no_grad():
            if model_type == "full":
                # Full参数微调模型的forward调用
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # LoRA微调模型的forward调用
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 统一处理输出格式
            if hasattr(outputs, 'logits'):
                next_token_logits = outputs.logits[:, -1, :]
            elif isinstance(outputs, dict) and 'logits' in outputs:
                next_token_logits = outputs['logits'][:, -1, :]
            else:
                raise ValueError(f"无法从模型输出中获取logits，输出类型: {type(outputs)}")
        
        # 应用temperature进行采样
        next_token_logits = next_token_logits / temperature
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        
        # 保存生成的token
        generated_tokens.append(next_token.item())
        
        # 检查是否遇到结束token
        if next_token.item() == tokenizer.eos_token_id:
            print(f"达到结束token，生成在第{step+1}步结束")
            break
        
        # 将新token添加到序列
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token, device=attention_mask.device)], dim=-1)
    
    # 构建完整的生成文本
    full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return full_generated_text

def main():
    args = parse_args()
    
    # 根据模型类型加载相应的模型类
    print(f"正在从 {args.model_path} 加载模型，类型: {args.model_type}")
    
    if args.model_type == "lora":
        from lora_model_customization import HippoLoRAQwen
        model = HippoLoRAQwen.from_pretrained(args.model_path)
    else:
        from model_customization import ModifiedQwen
        model = ModifiedQwen.from_pretrained(args.model_path)
    
    model.eval()  # 切换到评估模式
    
    # 获取分词器
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
    else:
        # 如果模型没有tokenizer，尝试从基础模型获取
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        except:
            # 最后的fallback
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # 验证参数完整性（可选）
    if args.check_params:
        print("\n验证模型参数完整性...")
        verify_model_parameters(model)
        print("验证完成！")
    
    # 交互式模式或单次执行模式
    if args.prompt is not None:
        # 单次执行模式：使用命令行参数中的prompt
        process_single_prompt(args.prompt, model, tokenizer, args.max_new_tokens, args.temperature)
    else:
        # 交互式模式：从命令行实时输入
        print("\n进入交互式对话模式！")
        print("输入 'exit' 或 'quit' 退出程序")
        print("输入 'reset_hidden' 重置HIPPO模型隐藏状态")
        print("========================================")
        print("注意：系统通过HIPPO模型的隐藏状态自动维护对话上下文，无需额外存储历史记录")
        
        while True:
            # 获取用户输入
            try:
                user_input = input("\n用户: ")
                
                # 检查退出命令
                if user_input.lower() in ["exit", "quit"]:
                    print("\n感谢使用，再见！")
                    break
                
                # 检查重置隐藏状态命令
                if user_input.lower() == "reset_hidden":
                    model.hidden_h = model.hippo_model.reset_h(batch_size=1)
                    print("HIPPO模型隐藏状态已重置")
                    continue
                    
                # 直接使用用户输入，移除格式标记
                input_text = user_input
                
                # 使用自定义生成函数
                print("正在生成回复...")
                assistant_response = custom_generate(
                    model=model,
                    model_type=args.model_type,
                    tokenizer=tokenizer,
                    input_text=input_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    reset_state=False  # 保持隐藏状态，实现跨轮记忆
                )
                
                # 打印结果
                print(f"\n助手: {assistant_response}")
                
            except KeyboardInterrupt:
                print("\n\n程序已中断")
                break
            except Exception as e:
                print(f"\n处理时出错: {e}")
                import traceback
                traceback.print_exc()

def process_single_prompt(prompt, model, model_type, tokenizer, max_new_tokens, temperature):
    """处理单个提示词并生成回复"""
    # 构建输入 - 直接使用提示词，不添加任何格式标记
    input_text = prompt
    
    # 使用自定义生成函数
    # 单次执行模式下，重置隐藏状态确保独立性
    response = custom_generate(
        model=model,
        model_type=model_type,
        tokenizer=tokenizer,
        input_text=input_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        reset_state=True  # 单次执行模式下重置隐藏状态
    )
    
    # 打印结果
    print("生成结果:")
    print(response.strip())

def verify_model_parameters(model):
    """验证模型自定义参数是否已正确加载"""
    # 检查hippo_model参数
    if hasattr(model, 'hippo_model'):
        param_values = []
        for param in model.hippo_model.parameters():
            param_values.extend(param.data.cpu().numpy().flatten())
        # 检查参数是否全为零或初始随机值（简单验证）
        param_mean = sum(param_values) / len(param_values) if param_values else 0
        print(f"  - Hippo模型参数平均值: {param_mean:.6f}")
    
    # 检查gate_mechanisms参数
    if hasattr(model, 'gate_mechanisms'):
        total_gates = len(model.gate_mechanisms)
        non_zero_gates = 0
        for gate_name in model.gate_mechanisms:
            gate = model.gate_mechanisms[gate_name]
            for param in gate.parameters():
                if not torch.all(param.data == 0):
                    non_zero_gates += 1
                    break
        print(f"  - 门控机制参数状态: {non_zero_gates}/{total_gates} 个门控有非零参数")
    
    # 检查融合层配置
    if hasattr(model, 'fusion_layers'):
        print(f"  - 融合层配置: {model.fusion_layers}")
    
    # 检查Hippo矩阵缩放因子
    if hasattr(model, 'hippo_model') and hasattr(model.hippo_model, 'hippo_scale'):
        print(f"  - Hippo矩阵缩放因子: {model.hippo_model.hippo_scale}")

if __name__ == "__main__":
    main()
    