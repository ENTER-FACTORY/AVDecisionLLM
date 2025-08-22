#!/usr/bin/env python3
"""
python /home/haibenben/waz/trl/examples/scripts/merge_lora_weights.py \
    --base_model_path /kpfs/model/Qwen2.5/Qwen2.5-32B-Instruct \
    --lora_weights_path /home/haibenben/waz/trl/Qwen2.5_32B_sft/lora_weights \
    --output_path ./merged_model_clean
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import json
import shutil


def get_model_files_info(model_path):
    """获取模型文件信息"""
    files = []
    total_size = 0
    
    for file in os.listdir(model_path):
        if file.endswith(('.safetensors', '.bin')):
            file_path = os.path.join(model_path, file)
            size = os.path.getsize(file_path)
            files.append((file, size))
            total_size += size
    
    return sorted(files), total_size


def fixed_merge_lora_weights(base_model_path, lora_weights_path, output_path):
    """
    修复版合并函数 - 确保文件数量不增加
    """
    print("=== 修复版LoRA合并 ===")
    
    # 检查原始模型信息
    print("1. 检查原始模型...")
    base_files, base_size = get_model_files_info(base_model_path)
    print(f"原始模型文件数: {len(base_files)}")
    print(f"原始模型大小: {base_size / (1024**3):.2f} GB")
    
    # 加载LoRA配置
    print("2. 检查LoRA配置...")
    peft_config = PeftConfig.from_pretrained(lora_weights_path)
    print(f"LoRA rank: {peft_config.r}, alpha: {peft_config.lora_alpha}")
    
    # 加载基础模型
    print("3. 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # 加载LoRA权重
    print("4. 加载LoRA权重...")
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        torch_dtype=torch.bfloat16
    )
    
    # 合并权重
    print("5. 合并权重...")
    merged_model = model_with_lora.merge_and_unload()
    
    # 检查合并是否成功
    print("6. 验证合并...")
    original_param_count = sum(p.numel() for p in base_model.parameters())
    merged_param_count = sum(p.numel() for p in merged_model.parameters())
    
    if original_param_count != merged_param_count:
        raise ValueError(f"参数数量不匹配！原始: {original_param_count}, 合并后: {merged_param_count}")
    
    print(f"✅ 参数数量匹配: {merged_param_count:,}")
    
    # 创建输出目录
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    # 保存合并后的模型（使用与原始模型相同的保存策略）
    print("7. 保存模型...")
    
    # 获取原始模型的index文件来确定分片策略
    original_index_file = os.path.join(base_model_path, "model.safetensors.index.json")
    if os.path.exists(original_index_file):
        with open(original_index_file, 'r') as f:
            original_index = json.load(f)
        max_shard_size = original_index.get('metadata', {}).get('total_size', '10GB')
        
        # 如果没有明确的分片大小，根据文件数量估算
        if isinstance(max_shard_size, str) and 'GB' in max_shard_size:
            estimated_shard_size = max_shard_size
        else:
            # 估算每个分片的大小
            estimated_shard_size = f"{int(base_size / len(base_files) / (1024**3)) + 1}GB"
    else:
        # 如果是单文件模型
        estimated_shard_size = "10GB"
    
    print(f"使用分片大小: {estimated_shard_size}")
    
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=estimated_shard_size
    )
    
    # 保存tokenizer
    print("8. 保存tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    
    # 检查输出文件
    print("9. 检查输出...")
    output_files, output_size = get_model_files_info(output_path)
    print(f"输出文件数: {len(output_files)}")
    print(f"输出大小: {output_size / (1024**3):.2f} GB")
    
    if len(output_files) > len(base_files) * 1.5:  # 允许一些合理的差异
        print(f"⚠️ 警告: 输出文件数量异常增加！")
        print("原始文件:")
        for name, size in base_files[:5]:  # 显示前5个
            print(f"  {name}: {size / (1024**2):.1f} MB")
        print("输出文件:")
        for name, size in output_files[:5]:  # 显示前5个
            print(f"  {name}: {size / (1024**2):.1f} MB")
    else:
        print("✅ 文件数量正常")
    
    # 简单功能测试
    print("10. 功能测试...")
    try:
        test_model = AutoModelForCausalLM.from_pretrained(
            output_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="gpu"
        )
        
        test_tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True)
        inputs = test_tokenizer("Hello", return_tensors="pt")
        
        with torch.no_grad():
            outputs = test_model.generate(**inputs, max_new_tokens=5)
        
        result = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"测试输出: {result}")
        print("✅ 功能测试通过")
        
        del test_model, test_tokenizer
        
    except Exception as e:
        print(f"⚠️ 功能测试失败: {e}")
    
    # 清理内存
    del base_model, model_with_lora, merged_model
    torch.cuda.empty_cache()
    
    print("=== 合并完成 ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--lora_weights_path", required=True)
    parser.add_argument("--output_path", required=True)
    
    args = parser.parse_args()
    
    fixed_merge_lora_weights(
        args.base_model_path,
        args.lora_weights_path,
        args.output_path
    )


if __name__ == "__main__":
    main()

# 拉起推理服务进行测试
'''
vllm serve /home/haibenben/waz/trl/merged_model_clean \
    --host 0.0.0.0 \
    --port 8000 \
'''

'''

# 发起
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/haibenben/waz/trl/merged_model_clean",
    "messages": [{"role": "user", "content": "Scenario ID: 14832\n\nCurrent driving scenario analysis: Autonomous vehicle status: AV current position: origin (0.0, 0.0) (using step 50 position as coordinate origin), heading angle: 2.13 radians, current speed: 0.0 m/s. Distance to stop line: 4.9 meters. Traffic signal status: Traffic signal: green light, 8 seconds remaining. Position relative to intersection: not inside intersection. Intersection approach status: moving away from intersection. Lane position: The autonomous vehicle is currently in a middle lane. Lane configuration: Current road has 1 lanes in total, 1 straight lanes. Road centerline information: Lane 1 (straight) (relative coordinates): [(0.0, 0.0), (-0.0, -0.1), (-0.1, -0.2), (-0.1, -0.3), (-0.2, -0.4), (-0.2, -0.4), (-0.3, -0.5), (-0.3, -0.6), (-0.4, -0.7), (-0.4, -0.8)]. Surrounding vehicles (within 50m): 5 vehicles detected. Vehicle 1 (type: CAR): current position (-6.1, -5.5), heading 2.14 radians, speed 3.0 m/s, future trajectory points (steps 1,10,20,30,40,50): (-6.2, -5.8), (-7.2, -7.8), (-8.0, -9.4), (-8.7, -10.3), (-8.6, -10.3), (-8.6, -10.3) Vehicle 2 (type: CAR): current position (7.3, 7.1), heading 2.14 radians, speed 3.3 m/s, future trajectory points (steps 1,10,20,30,40,50): (7.2, 6.9), (5.8, 4.3), (5.2, 2.5), (4.3, 1.1), (4.1, 0.4), (3.8, -0.0) Vehicle 3 (type: CAR): current position (-7.6, -35.5), heading 1.11 radians, speed 4.0 m/s, future trajectory points (steps 1,10,20,30,40,50): (-8.1, -35.7), (-11.8, -37.4), (-16.1, -40.8), (-20.1, -45.8), (-24.2, -52.4), (-27.7, -60.0) Vehicle 4 (type: CAR): current position (15.2, 37.8), heading 2.11 radians, speed 10.6 m/s, future trajectory points (steps 1,10,20,30,40,50): (14.5, 36.7), (10.6, 28.7), (6.3, 20.5), (3.0, 13.6), (0.1, 7.8), (-1.9, 3.4) Vehicle 5 (type: CAR): current position (-6.4, 15.4), heading 2.10 radians, speed 10.4 m/s, future trajectory points (steps 1,10,20,30,40,50): (-7.0, 14.4), (-11.0, 6.7), (-15.7, -0.9), (-21.1, -6.4), (-27.5, -9.4), (-34.7, -10.1). Based on the above environmental information, what driving behavior should the autonomous vehicle choose?"}],
    "max_tokens": 1024
  }'
'''
