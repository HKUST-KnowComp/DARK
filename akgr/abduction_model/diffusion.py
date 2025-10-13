import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import re

from akgr.abduction_model.gpt2_dit import DiffusionModel
from akgr.abduction_model.transformer import create_transformer
from typing import Optional
from akgr.utils.diffusion_utils import sample_tokens

class Diffusion(nn.Module):
    def __init__(
        self,
        ntoken,
        special_tokens,
        model_name,
        config_model,
        device, drop, generation_config):
        super().__init__()

        self.device = device
        self.mask_index = ntoken
        base_model = create_transformer(
            ntoken=ntoken,
            special_tokens=special_tokens,
            model_name=model_name,
            config_model=config_model
        ).to(device)
        self.model = DiffusionModel(base_model)
        self.drop = drop
        self.generation_config = generation_config
        # 简化的时间步参数
        self.num_timesteps = 500
        
        # 约束相关的token映射
        self.special_tokens = special_tokens
        self.token_to_id = {v: k for k, v in special_tokens.items()}
        self.id_to_token = special_tokens
        self.config = base_model.config
        # 逻辑操作符
        self.logic_operators = ['i', 'u', 'n', 'p', 'e']
        self.logic_operator_ids = [special_tokens.get(op, -1) for op in self.logic_operators]
    
    def forward_process(self, batch, eps=1e-3):
        """简化的前向过程"""
        b, l = batch.shape
        
        t = torch.rand((b,), device=batch.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        
        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        noisy_batch = torch.where(mask_indices, self.mask_index, batch)
        
        return noisy_batch, p_mask, t
    
    def training_step(self, inputs, attention_mask, weight_attention=None, mode='unify', attention_weight=False):
        """简化的训练步骤"""
        x = inputs.detach()
        b, l = inputs.shape
        # 使用简化的前向过程
        noisy_input, p_mask, t = self.forward_process(x)
        if mode == 'sft':
            # 保持source部分不变
            noisy_input[attention_mask == 1] = inputs[attention_mask == 1].clone()
           
        
        mask_indices = (noisy_input == self.mask_index)
        
        
        # 获取模型预测
        logits = self.model(noisy_input)
        
        # 交叉熵损失
        ce_loss = F.cross_entropy(
                logits[mask_indices], 
                inputs[mask_indices], 
                reduction="none"
            ).float()
            
        # 简化的时间权重
        time_weight = 1.0 / (p_mask[mask_indices])
        
        # 注意力权重调整
        if attention_weight:
            # 获取mask位置的source_attention_mask
            if mode == 'unify':
                mask_source_attention = attention_mask[mask_indices]
            elif mode == 'sft':
                mask_source_attention = weight_attention[mask_indices]
            # source部分的loss权重为2，其他部分为1
            attention_weight_tensor = torch.where(mask_source_attention == 1, 5.0, 1.0)
            # 应用注意力权重
            weighted_loss = ce_loss * time_weight * attention_weight_tensor
        else:
            weighted_loss = ce_loss * time_weight
            
        # 最终损失
        loss = weighted_loss.sum() / mask_indices.sum()
        
        
        return loss, logits
    
    def add_gumbel_noise(self, logits, temperature):
        """简化的Gumbel噪声"""
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-8) + 1e-8)
        return logits + gumbel_noise * temperature

    def glumel_noise(self, logits, tau):
        return F.gumbel_softmax(logits, tau=tau, hard=False)

    
    
    
    
    def compute_kl(self, logits1, logits2):
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.softmax(logits2, dim=-1)
        #kl(p2,p1)
        return F.kl_div(p1, p2, reduction='none').sum(dim=-1)
    
    def forward(self, inputs, attention_mask=None):
        return self.model(inputs, attention_mask)
        
    # @torch.no_grad()
    # def diff_sample(self, prompt=None, batch_size=1, alg='greedy', steps=512, temperature=0.1, 
    #                cfg_scale=0., context_length=2048, eps=1e-5, device='cuda', source_attention=None):
    #     """简化的采样过程"""
        
    #     batch_size = batch_size if prompt is None else prompt.shape[0]
    #     x = torch.full((batch_size, context_length), self.mask_index, dtype=torch.long).to(device)
    #     x[:, :prompt.shape[1]] = prompt.clone()
    #     src_tgt_attention = torch.ones((batch_size, context_length)).to(device)
    #     src_tgt_attention[:, :prompt.shape[1]] = source_attention.clone()
        
    #     x_save = []
    #     x_save.append(x.clone())
        
    #     # 简化的时间步调度
    #     timesteps = torch.linspace(self.num_timesteps - 1, 0, steps + 1, device=device).long()
        
    #     for i in range(steps):
            
    #         mask_index = (x == self.mask_index)
    #         if mask_index.sum() == 0:
    #             break
            
    #         with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #             if cfg_scale > 0.:
    #                 # Classifier-free guidance
    #                 un_x = x.clone()
    #                 un_x[:, :prompt.shape[1]] = self.mask_index
    #                 x_ = torch.cat([x, un_x], dim=0)
    #                 un_src_tgt_attention = src_tgt_attention.clone()
    #                 attn = torch.cat([src_tgt_attention, un_src_tgt_attention], dim=0)
                    
    #                 logits = self.model(x_, attn)
    #                 logits, un_logits = torch.chunk(logits, 2, dim=0)
    #                 logits, un_logits = logits[mask_index], un_logits[mask_index]
    #             else:
    #                 logits = self.model(x, src_tgt_attention)[mask_index]
            
    #         if cfg_scale > 0.:
    #             logits = un_logits + cfg_scale * (logits - un_logits)
            
    #         t = timesteps[i]
    #         s = timesteps[i + 1] if i < steps - 1 else torch.tensor(0, device=device)
            
    #         if alg == 'origin':
    #             # 简化的原始算法
    #             p_transfer = 1 - s.float() / t.float() if i < steps - 1 else 1.0
    #             x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + self.mask_index
                
    #             transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
    #             if transfer_index_t_s.sum() > 0:
    #                 logits_with_noise = self.add_gumbel_noise(
    #                     logits[transfer_index_t_s], temperature=temperature
    #                 )
    #                 x0[transfer_index_t_s] = torch.argmax(logits_with_noise, dim=-1)
                
    #             x[mask_index] = x0.clone()
    #             x_save.append(x.clone())
                
    #         elif alg == 'greedy':
    #             # 简化的贪婪算法
    #             logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
    #             x0 = torch.argmax(logits_with_noise, dim=-1)
                
    #             # 计算置信度
    #             probs = F.softmax(logits, dim=-1)
    #             confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                
    #             num_mask_token = mask_index.sum()
    #             number_transfer_tokens = int(num_mask_token * (1 - s.float() / t.float())) if i < steps - 1 else num_mask_token
                
    #             if number_transfer_tokens > 0:
    #                 _, transfer_index = torch.topk(confidence, min(number_transfer_tokens, confidence.size(0)))
    #                 x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + self.mask_index
    #                 x0_[transfer_index] = x0[transfer_index].clone()
    #                 x[mask_index] = x0_
                    
    #         else:
    #             raise NotImplementedError(f"Algorithm {alg} not implemented")
        
    #     return x, 
    
    # @torch.no_grad()
    # def diff_sample_unify(self, prompt=None, batch_size=1, alg='origin', steps=512, temperature=0.1, 
    #                cfg_scale=0., context_length=2048, eps=1e-5, device='cuda', source_attention=None):
    #     """简化的采样过程"""
        
    #     x = torch.full(prompt.shape, self.mask_index, dtype=torch.long).to(device)
    #     x[source_attention == 1] = prompt[source_attention == 1].clone()
    #     src_tgt_attention = torch.ones(prompt.shape).to(device)
    #     src_tgt_attention[source_attention == 1] = source_attention[source_attention == 1]
    #     x_save = []
    #     x_save.append(x.clone())
        
    #     # 简化的时间步调度
    #     timesteps = torch.linspace(self.num_timesteps - 1, 0, steps + 1, device=device).long()
    #     for i in range(steps):
    #         mask_index = (x == self.mask_index)
    #         if mask_index.sum() == 0:
    #             break
            
    #         with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #             logits = self.model(x)[mask_index]
            
            
    #         t = timesteps[i]
    #         s = timesteps[i + 1] if i < steps - 1 else torch.tensor(0, device=device)
            
    #         if alg == 'origin':
    #             # 简化的原始算法
    #             p_transfer = 1 - s.float() / t.float() if i < steps - 1 else 1.0
    #             x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + self.mask_index
                
    #             transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
    #             if transfer_index_t_s.sum() > 0:
    #                 logits_with_noise = self.add_gumbel_noise(
    #                     logits[transfer_index_t_s], temperature=temperature
    #                 )
    #                 x0[transfer_index_t_s] = torch.argmax(logits_with_noise, dim=-1)
                
    #             x[mask_index] = x0.clone()
    #             x_save.append(x.clone())
                
    #         elif alg == 'greedy':
    #             # 简化的贪婪算法
    #             logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
    #             x0 = torch.argmax(logits_with_noise, dim=-1)
                
    #             # 计算置信度
    #             probs = F.softmax(logits, dim=-1)
    #             confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                
    #             num_mask_token = mask_index.sum()
    #             number_transfer_tokens = int(num_mask_token * (1 - s.float() / t.float())) if i < steps - 1 else num_mask_token
                
    #             if number_transfer_tokens > 0:
    #                 _, transfer_index = torch.topk(confidence, min(number_transfer_tokens, confidence.size(0)))
    #                 x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + self.mask_index
    #                 x0_[transfer_index] = x0[transfer_index].clone()
    #                 x[mask_index] = x0_
                    
    #         else:
    #             raise NotImplementedError(f"Algorithm {alg} not implemented")
        
    #     return x, x_save

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ):
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        mask_token_id = self.mask_index
        steps = generation_config.steps
        eps = 1e-12
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = []

        # pad input_ids to max_length
        
        x = torch.full(input_ids.shape, self.mask_index, dtype=torch.long).to(input_ids.device)
        x[attention_mask == 1] = input_ids[attention_mask == 1].clone()
        print(x[0])

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        for i in range(steps):
            mask_index = (x == mask_token_id)
            model_output = self.model(x)
            # TRL模型返回tuple，需要提取logits
            if isinstance(model_output, tuple):
                logits = model_output[0]  # 第一个元素是logits
            else:
                logits = model_output
            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices,transfer_index] = x_[row_indices,transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())
        
        if return_dict_in_generate:
            return {
                "sequences": x,
                "history": histories,
            }
        else:
            return x