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

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "transformers",
#     "datasets",
#     "torch",
#     "accelerate",
# ]
# ///

"""
训练本地LLM模型进行监督微调(SFT)

使用方法:
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml sft_local_llm.py \
    --model_path /path/to/your/local/model \
    --dataset_path /path/to/your/local/dataset \
    --output_dir ./output
"""

import argparse
import os
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer
import torch


def format_instruction_data(example):
    """
    将包含instruction, input, output的数据格式化为训练用的text格式
    
    Args:
        example: 包含'instruction', 'input', 'output'字段的数据样本
    
    Returns:
        格式化后的文本
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    # 构建对话格式的prompt
    if input_text.strip():
        # 如果有input字段，将instruction和input合并
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        # 如果没有input字段，只使用instruction
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    
    return {"text": prompt}


def load_local_dataset(dataset_path):
    """
    加载本地数据集
    
    Args:
        dataset_path: 数据集路径，支持json, jsonl, csv, parquet等格式
    
    Returns:
        处理后的数据集
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
        # 如果是目录，尝试加载目录下的所有支持格式的文件
        dataset = load_dataset(dataset_path, split='train')
    else:
        raise ValueError(f"不支持的数据集格式: {dataset_path}")
    
    # 格式化数据
    dataset = dataset.map(format_instruction_data, remove_columns=dataset.column_names)
    
    return dataset


def setup_model_and_tokenizer(model_path):
    """
    设置模型和tokenizer（为后续GRPO训练保持完整精度）
    
    Args:
        model_path: 本地模型路径
    
    Returns:
        model, tokenizer
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
    
    # 尝试使用Flash Attention 2，如果不可用则使用标准实现
    attn_implementation = "eager"  # 默认使用标准实现
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        print("使用 Flash Attention 2 加速训练")
    except ImportError:
        print("Flash Attention 2 未安装，使用标准注意力实现")
    
    # 加载模型（保持bf16精度，不使用量化）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # 使用bf16保持训练稳定性
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        device_map="auto",  # 自动分配设备
    )
    
    return model, tokenizer


def setup_lora_config(r=8, lora_alpha=16, lora_dropout=0.1):
    """
    设置LoRA配置（显存优化版本）
    """
    from peft import LoraConfig, TaskType
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,  # LoRA rank，更小的值需要更少显存
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    return lora_config


def main():
    parser = argparse.ArgumentParser(description="本地LLM模型SFT训练（为GRPO做准备）")
    parser.add_argument("--model_path", type=str, default="/kpfs/model/Qwen2.5/Qwen2.5-32B", help="本地模型路径")
    parser.add_argument("--dataset_path", type=str, default="/home/haibenben/waz/trl/data/v2x_seq_sft_thinking_builtin.json", help="本地数据集路径")
    parser.add_argument("--output_dir", type=str, default="./sft_output", help="输出目录")
    parser.add_argument("--use_lora", action="store_true", default=True, help="是否使用LoRA微调（显存不足时推荐）")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度（减少显存占用）")
    parser.add_argument("--batch_size", type=int, default=4, help="每设备批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数（增加有效批大小）")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率（LoRA用更大学习率）")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--save_steps", type=int, default=500, help="保存间隔步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估间隔步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    
    # 显存优化选项
    parser.add_argument("--use_cpu_offload", action="store_true", help="启用CPU卸载（DeepSpeed Zero3）")
    parser.add_argument("--use_8bit_optimizer", action="store_true", help="使用8bit优化器")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank（更小=更省显存）")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    args = parser.parse_args()
    
    print("=== 显存优化配置 ===")
    print(f"使用LoRA: {args.use_lora}")
    print(f"序列长度: {args.max_length}")
    print(f"批大小: {args.batch_size}")
    print(f"梯度累积: {args.gradient_accumulation_steps}")
    print(f"LoRA rank: {args.lora_rank}")
    print("===================")
    
    print(f"加载数据集: {args.dataset_path}")
    # 加载数据集
    train_dataset = load_local_dataset(args.dataset_path)
    print(f"数据集大小: {len(train_dataset)}")
    print(f"数据样本示例:\n{train_dataset[0]['text'][:200]}...")
    
    print(f"加载模型: {args.model_path}")
    # 加载模型和tokenizer（不使用量化，为GRPO训练做准备）
    model, tokenizer = setup_model_and_tokenizer(args.model_path)
    
    # 设置训练配置（参照原始例子，只保留核心参数）
    training_args = SFTConfig(
        output_dir=args.output_dir,
        bf16=True,  # 保持bf16，与后续GRPO训练兼容且节省显存
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
        logging_dir=f"{args.output_dir}/logs",
        
        # 显存优化配置
        dataloader_pin_memory=False,  # 不固定内存
        remove_unused_columns=True,   # 移除未使用的列
        optim="paged_adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",
    )
    
    # 如果使用LoRA（强烈推荐用于显存不足的情况）
    peft_config = None
    if args.use_lora:
        peft_config = setup_lora_config(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        print(f"使用LoRA微调（rank={args.lora_rank}, alpha={args.lora_alpha}）")
    else:
        print("使用全参数微调（显存需求很大）")
    
    # 创建trainer（按照原始例子的简化参数）
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    
    # 开始训练
    print("开始SFT训练（为后续GRPO训练做准备）...")
    trainer.train()
    
    # 保存模型
    print(f"保存模型到: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # 如果使用了LoRA，也保存LoRA权重
    if args.use_lora:
        trainer.model.save_pretrained(f"{args.output_dir}/lora_weights")
        print("LoRA权重已保存，后续GRPO训练需要先合并权重")
    else:
        print("全参数模型已保存，可直接用于GRPO训练")
    
    print("SFT训练完成！模型已为GRPO训练做好准备。")


if __name__ == "__main__":
    main()
