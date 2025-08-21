#!/usr/bin/env python3
"""
合并LoRA权重到基础模型的脚本

使用方法:
python merge_lora_weights.py \
    --base_model_path /path/to/base/model \
    --lora_weights_path /path/to/lora/weights \
    --output_path /path/to/merged/model
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import json


def merge_lora_weights(base_model_path, lora_weights_path, output_path, device_map="auto"):
    """
    将LoRA权重合并到基础模型中
    
    Args:
        base_model_path: 基础模型路径
        lora_weights_path: LoRA权重路径
        output_path: 合并后模型的输出路径
        device_map: 设备映射策略
    """
    print(f"正在加载基础模型: {base_model_path}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    
    print(f"正在加载LoRA权重: {lora_weights_path}")
    
    # 加载LoRA配置
    peft_config = PeftConfig.from_pretrained(lora_weights_path)
    
    # 创建PEFT模型（加载LoRA权重）
    model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    
    print("正在合并LoRA权重...")
    
    # 合并权重并卸载LoRA层
    merged_model = model.merge_and_unload()
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    print(f"正在保存合并后的模型到: {output_path}")
    
    # 保存合并后的模型
    merged_model.save_pretrained(
        output_path,
        save_function=torch.save,
        safe_serialization=True
    )
    
    # 加载并保存tokenizer
    print("正在保存tokenizer...")
    try:
        # 首先尝试从LoRA权重目录加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            lora_weights_path,
            trust_remote_code=True
        )
    except:
        # 如果LoRA目录没有tokenizer，从基础模型加载
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
    
    tokenizer.save_pretrained(output_path)
    
    # 保存模型配置信息（移除了日期字段）
    config_info = {
        "base_model": base_model_path,
        "lora_weights": lora_weights_path,
        "torch_dtype": "bfloat16",
        "merged": True
    }
    
    with open(os.path.join(output_path, "merge_info.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    print("✅ LoRA权重合并完成！")
    print(f"合并后的模型已保存到: {output_path}")
    
    # 清理显存
    del base_model, model, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_merged_model(model_path):
    """
    验证合并后的模型是否可以正常加载
    
    Args:
        model_path: 合并后模型的路径
    """
    print(f"正在验证合并后的模型: {model_path}")
    
    try:
        # 尝试加载模型和tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu"  # 只用CPU验证，避免显存占用
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 简单的文本生成测试
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"测试生成: {generated_text}")
        print("✅ 模型验证通过！")
        
        # 清理
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ 模型验证失败: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="合并LoRA权重到基础模型")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        required=True,
        help="基础模型路径（原始预训练模型）"
    )
    parser.add_argument(
        "--lora_weights_path", 
        type=str, 
        required=True,
        help="LoRA权重路径（SFT训练输出的lora_weights目录）"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="合并后模型的输出路径"
    )
    parser.add_argument(
        "--device_map", 
        type=str, 
        default="auto",
        help="设备映射策略 (auto, cpu, cuda, 等)"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="是否验证合并后的模型"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="如果输出目录已存在，是否强制覆盖"
    )
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.base_model_path):
        raise ValueError(f"基础模型路径不存在: {args.base_model_path}")
    
    if not os.path.exists(args.lora_weights_path):
        raise ValueError(f"LoRA权重路径不存在: {args.lora_weights_path}")
    
    # 检查输出路径
    if os.path.exists(args.output_path) and not args.force:
        if os.listdir(args.output_path):  # 目录非空
            response = input(f"输出目录 {args.output_path} 已存在且非空，是否继续？(y/N): ")
            if response.lower() != 'y':
                print("操作已取消")
                return
    
    print("=== LoRA权重合并开始 ===")
    print(f"基础模型: {args.base_model_path}")
    print(f"LoRA权重: {args.lora_weights_path}")
    print(f"输出路径: {args.output_path}")
    print(f"设备映射: {args.device_map}")
    print("========================")
    
    # 执行合并
    try:
        merge_lora_weights(
            base_model_path=args.base_model_path,
            lora_weights_path=args.lora_weights_path,
            output_path=args.output_path,
            device_map=args.device_map
        )
        
        # 验证模型（可选）
        if args.verify:
            verify_merged_model(args.output_path)
            
    except Exception as e:
        print(f"❌ 合并过程中出现错误: {str(e)}")
        raise
    
    print("\n🎉 所有操作完成！合并后的模型可以用于GRPO训练或推理。")


if __name__ == "__main__":
    main()
