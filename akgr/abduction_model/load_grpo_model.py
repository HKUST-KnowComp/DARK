#!/usr/bin/env python3
"""
加载GRPO训练后的模型用于测试
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from akgr.abduction_model.diffusion import Diffusion
from akgr.abduction_model.transformer import create_transformer
from akgr.utils.load_util import load_yaml
from akgr.tokenizer import create_tokenizer
from akgr.dataloader import new_create_dataloader, new_create_dataset
from akgr.kgdata import load_kg

def load_grpo_model(checkpoint_path, device='cuda'):
    """
    加载GRPO训练后的模型 - 专为GPT2格式优化
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 设备
    
    Returns:
        state_dict: 直接返回state_dict（因为确保是GPT2格式）
        tokenizer: tokenizer
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # 检查检查点是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    
    # 直接加载权重文件
    state_dict_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"pytorch_model.bin not found in: {checkpoint_path}")
    
    print(f"Loading state dict from: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location=device)
    print(f"State dict loaded with {len(state_dict)} parameters")
    
    # 验证是GPT2格式
    gpt2_keys = [k for k in state_dict.keys() if 'transformer.' in k]
    opt_keys = [k for k in state_dict.keys() if 'decoder.' in k]
    
    print(f"Found {len(gpt2_keys)} GPT2-style keys, {len(opt_keys)} OPT-style keys")
    
    # 显示关键权重信息
    vocab_keys = []
    for key in state_dict.keys():
        if any(word in key.lower() for word in ['embed', 'wte', 'lm_head']):
            vocab_keys.append(key)
            print(f"  📊 {key}: {state_dict[key].shape}")
    
    if gpt2_keys:
        print("✅ Confirmed GPT2 format - returning state dict")
    else:
        print("⚠️  Warning: No GPT2-style keys found, but proceeding with state dict")
    
    return state_dict, tokenizer

def create_diffusion_model_with_loaded_weights(checkpoint_path, device='cuda'):
    """
    创建Diffusion模型并加载GRPO训练的权重
    
    Args:
        checkpoint_path: GRPO检查点路径
        device: 设备
    
    Returns:
        model: 加载了权重的Diffusion模型
        tokenizer: tokenizer
    """
    print(f"Creating Diffusion model and loading weights from: {checkpoint_path}")
    
    # 加载配置
    config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    config_model = load_yaml('akgr/configs/config-model.yml')
    
    # 加载知识图谱数据
    data_root = config_dataloader['data_root']
    dataname = config_dataloader['dataname']
    scale = config_dataloader['scale']
    max_answer_size = config_dataloader['max_answer_size']
    
    # 加载知识图谱
    graph_samplers, nentity, nrelation, offset, special_tokens = load_kg(
        data_root=data_root,
        dataname=dataname,
        scale=scale,
        max_answer_size=max_answer_size
    )
    
    # 创建tokenizer
    tokenizer, ntoken = create_tokenizer(
        special_tokens=special_tokens,
        offset=offset,
        nentity=nentity,
        nrelation=nrelation,
        is_gpt=True  # 假设是GPT模型
    )
    
    # 创建Diffusion模型
    model = Diffusion(
        ntoken=ntoken,
        special_tokens=special_tokens,
        model_name='gpt2',  # 根据实际情况调整
        config_model=config_model,
        device=device,
        drop=0.0,
        generation_config=None
    )
    
    # 加载GRPO训练的权重
    print("Loading GRPO weights...")
    grpo_model, _ = load_grpo_model(checkpoint_path, device)
    
    # 将GRPO模型的权重复制到Diffusion模型中
    # 这里需要根据实际的模型结构进行调整
    if hasattr(grpo_model, 'pretrained_model'):
        # TRL模型的情况
        grpo_state_dict = grpo_model.pretrained_model.state_dict()
    else:
        # 标准模型的情况
        grpo_state_dict = grpo_model.state_dict()
    
    # 过滤掉不匹配的键
    model_state_dict = model.model.state_dict()
    filtered_state_dict = {}
    
    for key, value in grpo_state_dict.items():
        if key in model_state_dict and model_state_dict[key].shape == value.shape:
            filtered_state_dict[key] = value
        else:
            print(f"Skipping key {key} due to shape mismatch or missing key")
    
    # 加载权重
    missing_keys, unexpected_keys = model.model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, graph_samplers

def main():
    parser = argparse.ArgumentParser(description='Load GRPO trained model for testing')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the GRPO checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to load model on')
    parser.add_argument('--mode', type=str, default='direct',
                       choices=['direct', 'diffusion'],
                       help='Loading mode: direct or diffusion')
    
    args = parser.parse_args()
    
    if args.mode == 'direct':
        # 直接加载GRPO模型
        model, tokenizer = load_grpo_model(args.checkpoint_path, args.device)
        print("Model loaded successfully for direct use")
        
        # 这里可以添加测试代码
        print("Model ready for testing!")
        
    elif args.mode == 'diffusion':
        # 创建Diffusion模型并加载权重
        model, tokenizer, graph_samplers = create_diffusion_model_with_loaded_weights(
            args.checkpoint_path, args.device
        )
        print("Diffusion model loaded successfully with GRPO weights")
        
        # 这里可以添加测试代码
        print("Model ready for testing!")

if __name__ == "__main__":
    main() 