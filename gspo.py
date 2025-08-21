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
#     "peft",
#     "datasets",
# ]
# ///

"""
Training script for autonomous driving decision making using GSPO.

使用方法:
1. 修改下面的 MODEL_PATH 和 DATASET_PATH 为你的实际路径
2. 直接运行: python gspo_autonomous_driving.py

或者使用accelerate加速训练:
accelerate launch gspo_autonomous_driving.py
"""

import torch
import json
import re
from datasets import Dataset
from sklearn.model_selection import train_test_split

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward


def load_local_dataset(dataset_path, test_size=0.1, random_state=42):
    """Load dataset from local JSON file and split into train/eval."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Split the data
    train_data, eval_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    return train_dataset, eval_dataset


if __name__ == "__main__":
    ################
    # 配置路径 - 请修改为你的实际路径
    ################
    MODEL_PATH = "/path/to/your/local/model"  # 修改为你的本地模型路径
    DATASET_PATH = "/path/to/your/dataset.json"  # 修改为你的数据集文件路径
    OUTPUT_DIR = "./gspo-autonomous-driving-output"  # 输出目录
    
    # 检查路径是否已经修改
    import os
    if MODEL_PATH == "/path/to/your/local/model" or DATASET_PATH == "/path/to/your/dataset.json":
        print("❌ 错误: 请先修改 MODEL_PATH 和 DATASET_PATH 为你的实际路径!")
        print("在文件开头找到以下行并修改:")
        print(f"MODEL_PATH = \"{MODEL_PATH}\"")
        print(f"DATASET_PATH = \"{DATASET_PATH}\"")
        exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型路径不存在: {MODEL_PATH}")
        exit(1)
        
    if not os.path.exists(DATASET_PATH):
        print(f"❌ 错误: 数据集文件不存在: {DATASET_PATH}")
        exit(1)
    
    print(f"✅ 模型路径: {MODEL_PATH}")
    print(f"✅ 数据集路径: {DATASET_PATH}")
    print(f"✅ 输出目录: {OUTPUT_DIR}")
    
    # 创建一个模拟的命令行参数对象
    import argparse
    import sys
    
    # 模拟命令行参数
    sys.argv = [
        "gspo_autonomous_driving.py",
        "--model_name_or_path", MODEL_PATH,
        "--output_dir", OUTPUT_DIR,
        "--learning_rate", "1e-5",
        "--torch_dtype", "bfloat16",
        "--max_prompt_length", "2048",
        "--max_completion_length", "512",
        "--use_peft",
        "--lora_target_modules", "q_proj", "v_proj",
        "--log_completions",
        "--per_device_train_batch_size", "4",
        "--num_generations", "8",
        "--importance_sampling_level", "sequence",
        "--epsilon", "3e-4",
        "--epsilon_high", "4e-4",
        "--beta", "0.0",
        "--loss_type", "grpo",
        "--gradient_accumulation_steps", "4",
        "--steps_per_generation", "8",
        "--num_train_epochs", "3",
        "--save_steps", "100",
        "--logging_steps", "10",
    ]
    
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # 使用我们定义的数据集路径
    dataset_path = DATASET_PATH
    
    ################
    # Model & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    ################
    # Dataset
    ################
    train_dataset, eval_dataset = load_local_dataset(dataset_path)

    SYSTEM_PROMPT = (
        "You are an autonomous vehicle AI assistant. Analyze the driving scenario and choose the appropriate "
        "driving behavior. Think step by step about the situation, considering traffic signals, surrounding "
        "vehicles, road conditions, and safety factors. Provide your reasoning within <think></think> tags, "
        "followed by your decision."
    )

    def make_conversation(example):
        """Convert dataset example to conversation format."""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["prompt"]},
            ],
            "answer": example["answer"]  # Keep the answer for reward calculation
        }

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)

    # Remove original columns except answer which we need for reward
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in ["prompt", "answer"]])
    eval_dataset = eval_dataset.remove_columns([col for col in eval_dataset.column_names if col not in ["prompt", "answer"]])

    ################
    # Reward Functions for Training
    ################
    def driving_behavior_reward(completions, answer: list[str], **kwargs):
        """
        Reward function that checks if the completion contains the correct driving behavior.
        
        Args:
            completions: List of model completions
            answer: List of ground truth answers
            
        Returns:
            List of rewards (1.0 for correct, 0.0 for incorrect)
        """
        # Define the key driving behaviors
        DRIVING_BEHAVIORS = [
            "stop_and_wait",
            "go_straight", 
            "turn_left",
            "turn_right"
        ]
        
        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        
        for content, ground_truth in zip(contents, answer):
            reward = 0.0
            
            # Convert to lowercase for case-insensitive matching
            content_lower = content.lower()
            ground_truth_lower = ground_truth.lower()
            
            # Extract the expected behavior from ground truth
            expected_behavior = None
            for behavior in DRIVING_BEHAVIORS:
                if behavior in ground_truth_lower:
                    expected_behavior = behavior
                    break
            
            # Check if the model output contains the expected behavior
            if expected_behavior and expected_behavior in content_lower:
                reward = 1.0
            else:
                # Alternative: check for any valid driving behavior even if not exact match
                # This provides partial credit for reasonable outputs
                for behavior in DRIVING_BEHAVIORS:
                    if behavior in content_lower:
                        # Give partial reward if it's a different but valid behavior
                        reward = 0.3
                        break
            
            rewards.append(reward)
        
        return rewards

    def comprehensive_driving_reward(completions, answer: list[str], **kwargs):
        """
        Enhanced reward function that considers both behavior matching and reasoning quality.
        """
        # Define the key driving behaviors
        DRIVING_BEHAVIORS = [
            "stop_and_wait",
            "go_straight", 
            "turn_left",
            "turn_right"
        ]
        
        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        
        for content, ground_truth in zip(contents, answer):
            total_reward = 0.0
            
            # Convert to lowercase for case-insensitive matching
            content_lower = content.lower()
            ground_truth_lower = ground_truth.lower()
            
            # 1. Behavior matching reward (main component)
            expected_behavior = None
            for behavior in DRIVING_BEHAVIORS:
                if behavior in ground_truth_lower:
                    expected_behavior = behavior
                    break
            
            if expected_behavior and expected_behavior in content_lower:
                total_reward += 1.0  # Full reward for correct behavior
            else:
                # Partial reward for any valid driving behavior
                for behavior in DRIVING_BEHAVIORS:
                    if behavior in content_lower:
                        total_reward += 0.3
                        break
            
            # 2. Reasoning quality bonus (additional component)
            reasoning_keywords = [
                "traffic", "signal", "vehicle", "safety", "distance", 
                "speed", "intersection", "lane", "collision", "brake"
            ]
            
            keyword_count = sum(1 for keyword in reasoning_keywords if keyword in content_lower)
            reasoning_bonus = min(keyword_count * 0.1, 0.5)  # Max 0.5 bonus
            total_reward += reasoning_bonus
            
            # 3. Structure bonus for using <think> tags
            if "<think>" in content_lower and "</think>" in content_lower:
                total_reward += 0.2
            
            rewards.append(min(total_reward, 2.0))  # Cap at 2.0 to prevent excessive rewards
        
        return rewards

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[
            think_format_reward,  # Original format reward
            driving_behavior_reward,  # Main behavior matching reward
            comprehensive_driving_reward  # Enhanced reward with reasoning quality
        ],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
