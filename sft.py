#!/usr/bin/env python3
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
改进版 - 训练本地LLM模型进行监督微调(SFT)
主要改进: 更合理的输出路径设置
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer
import torch
import json


def generate_output_path(base_dir, model_path, dataset_path, lora_rank=8, lora_alpha=16, custom_suffix=""):
    """
    生成合理的输出路径
    
    Args:
        base_dir: 基础输出目录
        model_path: 模型路径
        dataset_path: 数据集路径
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        custom_suffix: 自定义后缀
    
    Returns:
        生成的输出路径
    """
    # 获取时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取模型名称
    model_name = Path(model_path).name
    if model_name == ".":
        model_name = Path(model_path).resolve().name
    
    # 提取数据集名称
    dataset_name = Path(dataset_path).stem
    
    # 构建输出目录名称
    dir_components = [
        f"sft_{model_name}",
        f"data_{dataset_name}",
        f"lora_r{lora_rank}_a{lora_alpha}",
        timestamp
    ]
    
    if custom_suffix:
        dir_components.insert(-1, custom_suffix)
    
    output_dir_name = "_".join(dir_components)
    output_path = os.path.join(base_dir, output_dir_name)
    
    return output_path


def save_training_metadata(output_dir, args, model_path, dataset_path, dataset_size):
    """
    保存训练元数据，便于后续追踪
    """
    metadata = {
        "training_info": {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "dataset_path": dataset_path,
            "dataset_size": dataset_size,
            "output_dir": output_dir
        },
        "training_args": {
            "use_lora": args.use_lora,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio
        },
        "paths": {
            "base_model": model_path,
            "dataset": dataset_path,
            "lora_weights": os.path.join(output_dir, "lora_weights") if args.use_lora else None,
            "merged_model_command": f"python merge_lora_weights.py --base_model_path {model_path} --lora_weights_path {os.path.join(output_dir, 'lora_weights')} --output_path ./merged_{Path(output_dir).name}"
        }
    }
    
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📋 训练元数据已保存: {metadata_path}")
    return metadata_path


def format_instruction_data(example):
    """
    将包含instruction, input, output的数据格式化为训练用的text格式
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    # 构建对话格式的prompt
    if input_text.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    
    return {"text": prompt}


def load_local_dataset(dataset_path):
    """
    加载本地数据集
    """
    # 根据文件扩展名自动识别格式
    if dataset_path.endswith('.json'):
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    elif dataset_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    elif dataset_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=dataset_path, split='train')
    elif dataset_path.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=dataset_path, split='train')
    elif os.path.isdir(dataset_path):
        dataset = load_dataset(dataset_path, split='train')
    else:
        raise ValueError(f"不支持的数据集格式: {dataset_path}")
    
    # 格式化数据
    dataset = dataset.map(format_instruction_data, remove_columns=dataset.column_names)
    
    return dataset


def setup_model_and_tokenizer(model_path):
    """
    设置模型和tokenizer
    """
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # 尝试使用Flash Attention 2
    attn_implementation = "eager"
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        print("使用 Flash Attention 2 加速训练")
    except ImportError:
        print("Flash Attention 2 未安装，使用标准注意力实现")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    
    return model, tokenizer


def setup_lora_config(r=8, lora_alpha=16, lora_dropout=0.1):
    """
    设置LoRA配置
    """
    from peft import LoraConfig, TaskType
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    return lora_config


def save_lora_model_properly(trainer, output_dir, tokenizer, base_model_path):
    """
    正确保存LoRA模型
    """
    print(f"正确保存LoRA模型到: {output_dir}")
    
    # 创建lora_weights目录
    lora_weights_dir = os.path.join(output_dir, "lora_weights")
    os.makedirs(lora_weights_dir, exist_ok=True)
    
    # 保存LoRA适配器权重
    print("保存LoRA适配器权重...")
    trainer.model.save_pretrained(lora_weights_dir)
    
    # 保存tokenizer到主目录
    print("保存tokenizer...")
    tokenizer.save_pretrained(output_dir)
    
    # 验证保存结果
    adapter_config_path = os.path.join(lora_weights_dir, "adapter_config.json")
    adapter_model_files = [f for f in os.listdir(lora_weights_dir) if f.startswith('adapter_model')]
    
    if os.path.exists(adapter_config_path) and adapter_model_files:
        print("✅ LoRA权重保存成功")
        print(f"   配置文件: {adapter_config_path}")
        print(f"   权重文件: {adapter_model_files}")
    else:
        print("⚠️ LoRA权重保存可能有问题")
    
    return lora_weights_dir


def main():
    parser = argparse.ArgumentParser(description="改进版 - 本地LLM模型SFT训练")
    
    # 基础参数
    parser.add_argument("--model_path", type=str, default="/kpfs/model/Qwen2.5/Qwen2.5-32B-Instruct", help="本地模型路径")
    parser.add_argument("--dataset_path", type=str, default="/home/haibenben/waz/trl/data/v2x_seq_sft_thinking_builtin.json", help="本地数据集路径")
    
    # 输出路径相关参数
    parser.add_argument("--base_output_dir", type=str, default="./experiments", help="基础输出目录")
    parser.add_argument("--output_suffix", type=str, default="", help="输出目录自定义后缀")
    parser.add_argument("--custom_output_dir", type=str, default="./Qwen2.5_32B_sft", help="完全自定义输出目录（会覆盖自动生成的路径）")
    
    # 训练参数
    parser.add_argument("--use_lora", action="store_true", default=True, help="是否使用LoRA微调")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="每设备批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--save_steps", type=int, default=500, help="保存间隔步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估间隔步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # 优化选项
    parser.add_argument("--use_cpu_offload", action="store_true", help="启用CPU卸载")
    parser.add_argument("--use_8bit_optimizer", action="store_true", help="使用8bit优化器")
    
    args = parser.parse_args()
    
    # 生成或使用输出路径
    if args.custom_output_dir:
        output_dir = args.custom_output_dir
        print(f"使用自定义输出目录: {output_dir}")
    else:
        output_dir = generate_output_path(
            base_dir=args.base_output_dir,
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            custom_suffix=args.output_suffix
        )
        print(f"自动生成输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 训练配置 ===")
    print(f"模型路径: {args.model_path}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"输出目录: {output_dir}")
    print(f"使用LoRA: {args.use_lora}")
    print(f"LoRA配置: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"训练参数: epochs={args.num_epochs}, lr={args.learning_rate}, batch_size={args.batch_size}")
    print("==================")
    
    # 加载数据集
    print(f"📂 加载数据集: {args.dataset_path}")
    train_dataset = load_local_dataset(args.dataset_path)
    print(f"数据集大小: {len(train_dataset)}")
    print(f"数据样本示例:\n{train_dataset[0]['text'][:200]}...")
    
    # 加载模型和tokenizer
    print(f"🤖 加载模型: {args.model_path}")
    model, tokenizer = setup_model_and_tokenizer(args.model_path)
    
    # 保存训练元数据
    save_training_metadata(output_dir, args, args.model_path, args.dataset_path, len(train_dataset))
    
    # 设置训练配置
    training_args = SFTConfig(
        output_dir=output_dir,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataset_num_proc=4,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        logging_dir=f"{output_dir}/logs",
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        optim="paged_adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",
    )
    
    # 设置LoRA配置
    peft_config = None
    if args.use_lora:
        peft_config = setup_lora_config(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        print(f"✅ 使用LoRA微调（rank={args.lora_rank}, alpha={args.lora_alpha}）")
    else:
        print("⚠️ 使用全参数微调（显存需求很大）")
    
    # 创建trainer
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    
    # 开始训练
    print("🚀 开始SFT训练...")
    trainer.train()
    
    # 正确保存模型
    if args.use_lora:
        print("💾 保存LoRA模型...")
        lora_weights_dir = save_lora_model_properly(trainer, output_dir, tokenizer, args.model_path)
        
        print(f"\n✅ SFT训练完成！")
        print(f"📁 输出目录: {output_dir}")
        print(f"🔗 LoRA权重: {lora_weights_dir}")
        print(f"\n📋 下一步合并权重:")
        model_name = Path(args.model_path).name
        merged_dir = f"./merged_{Path(output_dir).name}"
        print(f"python fixed_merge_lora.py \\")
        print(f"    --base_model_path {args.model_path} \\")
        print(f"    --lora_weights_path {lora_weights_dir} \\")
        print(f"    --output_path {merged_dir}")
        
    else:
        print("💾 保存全参数模型...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"✅ 全参数模型已保存到: {output_dir}")
    
    print(f"\n🎉 训练完成！所有文件保存在: {output_dir}")


if __name__ == "__main__":
    main()
