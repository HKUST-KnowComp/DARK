from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import logging
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from verl.utils.model import compute_position_id_with_mask

class AbductionTrainer(Trainer):
    """Custom trainer for abduction model training"""
    
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.custom_processing_class = kwargs.pop('custom_processing_class', None)
        self.device = kwargs.pop('device', 'cuda')
        self.mask_token_id = self.custom_processing_class.mask_token_id
        self.token_reweighting = kwargs.pop('token_reweighting', True)
        self.time_reweighting = kwargs.pop('time_reweighting', "original")
        self.weight_eos = kwargs.pop('weight_eos', True)
        self.alpha = kwargs.pop('alpha', 0.25)
        self.gamma = kwargs.pop('gamma', 2.0)
        self.name = kwargs.pop('name', 'gpt2')
        self.merge_prob = kwargs.pop('merge_prob', 0.5)
        # 多卡训练支持 - 避免在初始化时同步，可能导致卡死
        self.is_multi_gpu = torch.cuda.device_count() > 1
        self.attention_all = kwargs.pop('attention_all', True)
        super().__init__(*args, **kwargs)
    
    def forward_process(self, batch, eps=1e-3):
        """简化的前向过程"""
        b, l = batch.shape
        
        t = torch.rand((b,), device=batch.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        noisy_batch = torch.where(mask_indices, self.mask_token_id, batch)
        return noisy_batch, p_mask

    def masking_merge_for_response(self, input_tokens, merge_prob=0.5, merge_schedule="dynamic_inverse", use_uniform_merge_prob=0.5):
        """
        The process is:
        1. Independently mask each token with probability sampling_ratio.
            If a token is masked it is replaced by "<mask>", otherwise it remains unchanged.
        2. Scan the masked sequence for adjacent "<mask>" tokens. Whenever found, with probability merge_prob:
                - Mark the first token's label as "<expand>" (indicating the head of a merged pair).
                - Modify the attention_mask so that the second token is not attended to (i.e. set its attention_mask to 0).
            Tokens that are not part of a merge or are not masked are labeled as "<nonexpand>".
        3. Compute position_ids such that effective tokens (attention_mask==1) receive sequential indices, 
            while merged-out tokens (attention_mask==0) receive a default position of 0.
        
        Parameters:
        input_tokens (torch.Tensor): The original sequence of tokens as a tensor.
        sampling_ratio (float): The independent probability a token is replaced with "<mask>".
        merge_prob (float): The probability that a pair of adjacent "<mask>" tokens are merged.
        
        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - final_tokens: The token tensor after independent masking (tokens remain as masked or original).
            - labels: A tensor (same length as tokens) with each token labeled as 1 ("<expand>") or 0 ("<nonexpand>").
            - attention_mask: A tensor of binary values (1 for effective tokens, 0 for merged tokens).
        """
        sampling_ratio = torch.rand(1).to(input_tokens.device)
        # Step 1: Sampling masks
        mask = torch.rand_like(input_tokens, dtype=torch.float) < sampling_ratio
        final_tokens = input_tokens.clone()
        final_tokens[mask] = self.custom_processing_class.mask_token_id

        eos_mask = input_tokens == self.custom_processing_class.eos_token_id
        final_tokens[eos_mask] = self.custom_processing_class.mask_token_id
        
        # Initialize labels and attention_mask
        labels = input_tokens.clone()
        attention_mask = torch.ones_like(input_tokens, dtype=torch.long)

        ## Step 2: Merge
        num_masked = mask.sum().item()

        if torch.rand(1).item() < use_uniform_merge_prob:
            merge_schedule = "static"
        if merge_schedule == "dynamic_inverse":
            dynamic_merge_prob = merge_prob * (1 - (num_masked / input_tokens.size(0))) 
        elif merge_schedule == "dynamic_proportional":
            dynamic_merge_prob = merge_prob * (num_masked / input_tokens.size(0))
        elif merge_schedule == "static":
            dynamic_merge_prob = merge_prob
        elif merge_schedule == "random":
            dynamic_merge_prob = torch.rand(1).item() * merge_prob
        elif merge_schedule == "full_random":
            # So we need to vary merge_prob to [0,1] to make the model more robust
            dynamic_merge_prob = torch.rand(1).clamp(0.0, 0.95)
        else:
            raise ValueError(f"Unknown merge schedule: {merge_schedule}")

        rand_values = torch.rand(len(final_tokens)-1)
        
        for i in range(len(final_tokens)-1):
            if input_tokens[i] == self.custom_processing_class.eos_token_id:
                break
            if (final_tokens[i] == self.custom_processing_class.mask_token_id and 
                final_tokens[i+1] == self.custom_processing_class.mask_token_id and ## adjacement MASK
                rand_values[i] < dynamic_merge_prob): ## merge
                labels[i] = self.custom_processing_class.expand_token_id
                attention_mask[i+1] = 0

        return final_tokens, labels, attention_mask, sampling_ratio

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Custom loss computation for different training modes"""
        try:
            # Extract inputs based on training mode
            input_ids = inputs['input_ids']
            need_to_mask = inputs['attention_mask'].bool()  # Convert to boolean tensor
            if self.attention_all:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = torch.zeros_like(input_ids)
            
            # Create full-size labels tensor that matches input_ids
            full_labels = input_ids.clone()
            
            mask_input, partial_labels, noisy_attention_mask, t = self.masking_merge_for_response(
                    input_ids[need_to_mask], merge_prob=self.merge_prob, merge_schedule="dynamic_inverse", use_uniform_merge_prob=0.5
                )
            input_ids[need_to_mask] = mask_input    
            full_labels[need_to_mask] = partial_labels
            attention_mask[need_to_mask] = noisy_attention_mask
          
            # Use the full-size labels
            labels = full_labels
            position_ids = compute_position_id_with_mask(attention_mask)

            loss_mask = (input_ids == self.custom_processing_class.mask_token_id) & (attention_mask == 1)
            
            if self.name!='gpt2':
                if attention_mask.dim() == 2:
                        # Input is (B, S) -> need to create pairwise mask (B, S, S)
                    attention_mask = torch.logical_and(
                                attention_mask.unsqueeze(1).unsqueeze(-2),  # (B, 1, S, 1)
                                attention_mask.unsqueeze(1).unsqueeze(-1)   # (B, 1, S, 1)
                            )  # Result: (B, 1, S, S)

                elif attention_mask.dim() == 3:
                        # Already (B, S, S), just add head dimension
                    attention_mask = attention_mask.unsqueeze(1)  # (B, 1, S, S)
            
                output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                        )
            else:
                output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
            
            logits = output.logits
            shift_logits = torch.cat(
                            [logits[:, 0:1], logits[:, :-1]], dim=1
                        ).contiguous()
            shift_labels = labels.contiguous()
            
            # 修复多卡训练问题：确保设备一致性
            if hasattr(model, 'module'):
                vocab_size = model.module.config.vocab_size
            else:
                vocab_size = model.config.vocab_size
            
            # 修复维度处理问题
            batch_size = shift_logits.size(0) * shift_logits.size(1)
            shift_logits = shift_logits.view(batch_size, vocab_size)
            shift_labels = shift_labels.view(batch_size)
            
            # 确保设备一致性，避免多卡同步问题
            if shift_labels.device != shift_logits.device:
                shift_labels = shift_labels.to(shift_logits.device)
            
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits, shift_labels)
            
            # 修复mask处理
            loss_mask = loss_mask.reshape(-1)
            if loss_mask.device != loss.device:
                loss_mask = loss_mask.to(loss.device)
            
            loss = loss.masked_fill(~loss_mask, 0)
            
            if self.token_reweighting:
                loss = (
                    self.alpha
                    * (1 - torch.exp(-loss)) ** self.gamma
                    * loss
                )

            if self.time_reweighting == "original":          
                weight = 1 / t[:, None].float().expand(labels.size())
            elif self.time_reweighting == "linear":
                weight = 1 - t.float().expand(labels.size())
            
            # 确保weight和loss在同一设备上
            if weight.device != loss.device:
                weight = weight.to(loss.device)
            
            loss = loss * weight.reshape(-1)
           
            if self.weight_eos :
                non_eos_mask = (shift_labels != self.custom_processing_class.eos_token_id) & loss_mask
                non_eos_loss = loss.clone()  
                non_eos_loss[~non_eos_mask] = 0  
                non_eos_count = non_eos_mask.sum().item() 
                non_eos_loss = non_eos_loss.sum()  

                       
                eos_mask = (shift_labels == self.custom_processing_class.eos_token_id) & loss_mask
                eos_loss = loss.clone()  
                eos_loss[~eos_mask] = 0  
                eos_count = eos_mask.sum().item()  
                eos_loss = eos_loss.sum() / eos_count  
                        
                loss = (non_eos_loss + eos_loss) / (non_eos_count + 1)  
            else:
                valid_token_this_rank = torch.sum(loss_mask)
      
                loss = torch.sum(loss) / valid_token_this_rank
            
            # 多卡训练支持
            if num_items_in_batch is not None and hasattr(self, 'accelerator'):
                if self.accelerator.num_processes > 1:
                    if hasattr(self, 'state') and self.state.global_step % 100 == 0:
                        print(f"Multi-device training: {self.accelerator.num_processes} devices, "
                              f"batch items: {num_items_in_batch}, loss: {loss.item():.4f}") 
            
            return (loss, logits) if return_outputs else loss
            
        except Exception as e:
            print(f"compute_loss error: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个默认loss避免训练中断
            if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                dummy_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
                return dummy_loss
            else:
                raise e
    
    def evaluation_step(self, model, inputs):
        """Custom evaluation step for validation"""
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss
    
    def get_optimizer(self):
        """重写获取优化器的方法"""
        return super().get_optimizer()
    
    def get_scheduler(self):
        """重写获取学习率调度器的方法"""
        return super().get_scheduler()
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """重写获取训练采样器的方法"""
        return super()._get_train_sampler()
    
    def get_train_dataloader(self) -> DataLoader:
        """重写获取训练数据加载器的方法"""
        return super().get_train_dataloader()
    
    def _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval):
        """重写内部训练循环"""
        return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
            
    
    


class AbductionTrainingArguments(TrainingArguments):
    """Custom training arguments for abduction model"""
    
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.training_mode = kwargs.pop('training_mode', 'unify')
        
        super().__init__(*args, **kwargs)
