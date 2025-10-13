import os, sys, argparse, warnings

import json
from networkx import optimize_graph_edit_distance
import yaml
import pandas as pd

import torch
from tqdm import tqdm

import random

# dataloader
from akgr.dataloader import new_create_dataloader, new_create_dataset
from akgr.tokenizer import create_tokenizer, new_extract_sample_to_device, unify_extract_sample_to_device

# transformer (huggingface)
from akgr.abduction_model.transformer import create_transformer
from akgr.abduction_model.diffusion import Diffusion
# utils
from akgr.utils.stat_util import stat_scores_by_pattern#, initialize_scores_stat
from akgr.utils.load_util import load_yaml, load_model, save_model, load_and_filter_query_patterns
from akgr.kgdata import load_kg
import pandas as pd
from akgr.abduction_model.gpt2_dit import DiffusionModel
# evaluation
from akgr.evaluation import scoring_input_wordlist_batch, scoring_input_act_batch
from akgr.utils.parsing_util import qry_actionprefix_get_branching, is_strint
from akgr.abduction_model.load_grpo_model import load_grpo_model
import wandb
# from accelerate import Accelerator
from accelerate import Accelerator
import logging





raw_dataset = None

# global var
device = None
pattern_filtered = None

nentity = None
nrelation = None
offset = None
special_tokens = None

do_correction = False
graph_samplers = None
rl_scoring_list = []
rl_factor = []
rl_search_split = None


def train_loop(dataloader, model, tokenizer, optimizer, scheduler, model_name,
               is_gpt, is_act, src_len, tgt_len, accelerator, mode='unify'):
    # https://pytorch.org/docs/stable/optim.html
    model.train()
    niter = len(dataloader)
    total_loss = 0

    for iter, sample in (pbar := tqdm(enumerate(dataloader), total=niter)):
        # a list of tensors
        # input_ids: [src+sep+tgt]
        # source_attention_mask: source + sep 
        # target_attention_mask: sep + target
        source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_attention_mask= \
            unify_extract_sample_to_device(device, sample, tokenizer, is_gpt, src_len, tgt_len, False)
       
        optimizer.zero_grad()
        # attention_mask: where to unchange for sft.
        #sft_target: target as prompt
        #sft_source: source as prompt
        if mode == 'unify':
            loss, logits = model.training_step(inputs = input_ids,  attention_mask = attention_mask, mode='unify', attention_weight=True)
        elif mode == 'sft_target':
            loss, logits = model.training_step(inputs = input_ids,  attention_mask = target_attention_mask, weight_attention = attention_mask, mode='sft', attention_weight=True)
        elif mode == 'sft_source':
            loss, logits = model.training_step(inputs = input_ids,  attention_mask = source_attention_mask, weight_attention = attention_mask, mode='sft', attention_weight=True)
        _loss = loss.detach().cpu().numpy()
        

        pbar.set_description(f'loss: {_loss}')
        total_loss += _loss

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if not ('-5' in model_name or '-01' in model_name):
            scheduler.step()
    return total_loss / niter



def mask_source(device, source_attention_mask, pred, tokenizer):
    # print('source mask')
    # print(source_attention_mask[:3, :15])
    B = pred.shape[0]
    diff = pred.shape[-1] - source_attention_mask.shape[-1]
    prefix_mask = torch.cat([
        source_attention_mask,
        torch.zeros((B, diff), dtype=torch.bool, device=device)], dim=1).to(device)
    # print('prefix mask')
    # print(prefix_mask[:3, :15])
    pred[prefix_mask == 1] = tokenizer.pad_token_id

def valid_loop(args, dataloader, model, tokenizer, graph_samplers,
               is_gpt, is_act, src_len, tgt_len, verbose):
    model.eval()
    niter = len(dataloader)
    total_loss = 0

    # initialization
    # scores_stat = initialize_scores_stat(pattern_filtered)
    scores_all = []
    pattern_id_all = []


    with torch.no_grad():
        for iter, sample in (pbar := tqdm(enumerate(dataloader, start=1), total=niter)):
            source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
                new_extract_sample_to_device(device, sample, tokenizer, is_gpt, src_len, tgt_len, True)

            # # print('src, tgt shapes:', src.shape, tgt.shape)
            # loss, logits = model.training_step(inputs = input_ids, attention_mask = attention_mask, source_attention_mask = source_attention_mask)
            
            # # print(outputs)
            # pred_argmax = logits.argmax(2)

            pred = model.diff_sample(prompt=input_ids, 
                context_length= src_len+tgt_len,  device=device, source_attention = source_attention_mask)
            
            
            if is_gpt: mask_source(device, source_attention_mask, pred, tokenizer)
            pred_decoded = tokenizer.batch_decode(pred, skip_special_tokens=True)

            print('target (label)')
            print(target[:5])
            # print('input_ids')
            # print(input_ids)
            print('pred_de')
            print(pred_decoded[:5])
            
            scoring_fn = scoring_input_act_batch if is_act else scoring_input_wordlist_batch
            scores = scoring_fn(
                pred_word_batch=pred_decoded,
                label_word_batch=target,
                ans_word_batch=source,
                scoring_method=['smatch'],
                do_correction=args.do_correction,
                graph_samplers=graph_samplers,
                verbose=args.vs)
            print(scores)
            scores_all.extend(scores)
            pattern_id_all.extend(pattern_id)
            score_df = stat_scores_by_pattern(scores, pattern_id, pattern_filtered)

            # 注意：这里loss没有定义，因为我们没有调用training_step
            # 如果需要loss，可以取消注释下面的代码
            # loss, logits = model.training_step(inputs=input_ids, source_attention_mask=source_attention_mask)
            # _loss = loss.detach().cpu().numpy()
            # total_loss += _loss
            # pbar.set_description(f'loss ave: {total_loss/iter}, s: {round(score_df.loc["all", ("smatch","mean")], 4)}')
            
            pbar.set_description(f's: {round(score_df.loc["all", ("smatch","mean")], 4)}')
        # scores_ave = scores_sum / scores_cnt
    # 由于没有计算loss，返回0作为占位符
    return 0.0, score_df

def handle_epoch_end(epoch, loss_train, loss_valid, score_df, mode, result_path, loss_log, args, model, optimizer, scheduler, model_name, nepoch):
    # 日志
    if hasattr(args, 'do_valid') and args.do_valid and score_df is not None:
        msg = f'[{mode}] epoch: {epoch}, train loss: {loss_train}, valid loss: {loss_valid}, s (valid): {score_df.loc["all",("smatch","mean")]}'
    else:
        msg = f'[{mode}] epoch: {epoch}, train loss: {loss_train}'
    logging.info(msg)
    with open(result_path, 'a') as result_file:
        result_file.write(msg + '\n')

    # 保存模型 - 根据mode确定使用哪个loss_log key
    if mode == 'unify':
        valid_key = 'valid'
    else:  # sft mode
        valid_key = f'valid_{mode}'
    
    # 保存模型
    if epoch % args.save_frequency == 0 \
        or epoch == nepoch \
        or (hasattr(args, 'do_valid') and args.do_valid and len(loss_log[valid_key].values()) > 0 and loss_valid <= min(loss_log[valid_key].values())):
        ckpt_path = os.path.join(args.checkpoint_root, args.modelname,
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{mode}-2-{epoch}.pth')
        save_model(ckpt_path, 'model', model, optimizer, scheduler, epoch, loss_log)

    # 保存分数
    if epoch % args.save_frequency == 0 or epoch == nepoch:
        if hasattr(args, 'do_valid') and args.do_valid and score_df is not None:
            scores_path = os.path.join(args.result_root, args.modelname,
                f'{args.dataname}-{args.scale}-{args.max_answer_size}-{mode}-{epoch}-scores.csv')
            score_df.to_csv(scores_path)
    print('=' * 50)

# 主训练流程
def fit(args, nepoch, dataloader, model, tokenizer, optimizer, scheduler, graph_samplers,
        model_name, is_gpt, is_act, src_len, tgt_len,
        last_epoch, loss_log, verbose, accelerator, training_mode):
    if not accelerator is None:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
    result_path = os.path.join(args.result_root, args.modelname,
        f'{args.dataname}-{args.scale}-{args.max_answer_size}_mdm_results.txt')
    
    # 可选的初始验证
    if hasattr(args, 'do_initial_valid') and args.do_initial_valid:
        loss_valid, score_df = valid_loop(args,
                dataloader['valid'], model,
                tokenizer,
                graph_samplers,
                is_gpt, is_act, src_len, tgt_len,
                verbose)
    else:
        loss_valid = float('inf')
        score_df = None
    if training_mode == 'unify':
        loss_train = train_loop(
            dataloader['train'],
            model,
            tokenizer,
            optimizer, scheduler,
            model_name, is_gpt, is_act, src_len, tgt_len,
            accelerator, mode='unify')
        for epoch in range(last_epoch+1, nepoch+1): # epoch starts from 1
            print('lr:', scheduler.get_last_lr())
            loss_train = train_loop(
                dataloader['train'],
                model,
                tokenizer,
                optimizer, scheduler,
                model_name, is_gpt, is_act, src_len, tgt_len,
                accelerator, mode='unify')
            
            # 可选的验证循环
            if hasattr(args, 'do_valid') and args.do_valid:
                loss_valid, score_df = valid_loop(args,
                    dataloader['valid'], model,
                    tokenizer,
                    graph_samplers,
                    is_gpt, is_act, src_len, tgt_len,
                    verbose)
            else:
                loss_valid = float('inf')
                score_df = None
                
            if ('-5' in model_name or '-01' in model_name):
                scheduler.step()
            # exit()
            loss_log['train'][epoch] = loss_train
            loss_log['valid'][epoch] = loss_valid
            handle_epoch_end(epoch, loss_train, loss_valid, score_df, 'unify', result_path, loss_log, args, model, optimizer, scheduler, model_name, nepoch)
    else:  # sft
        # 初始化sft模式的loss_log结构
        for sft_mode in ['sft_target']:
            if f'train_{sft_mode}' not in loss_log:
                loss_log[f'train_{sft_mode}'] = {}
            if f'valid_{sft_mode}' not in loss_log:
                loss_log[f'valid_{sft_mode}'] = {}
        
        # 初始训练
        for sft_mode in ['sft_target']:
            loss_train = train_loop(
                dataloader['train'],
                model,
                tokenizer,
                optimizer, scheduler,
                model_name, is_gpt, is_act, src_len, tgt_len,
                accelerator, mode=sft_mode)
                
        # 每个epoch内对两个sft_mode都进行训练
        for epoch in range(last_epoch+1, nepoch+1):
            print('lr:', scheduler.get_last_lr())
            
            # 对每个sft_mode进行训练
            for sft_mode in ['sft_target', 'sft_source']:
                print(f'Training {sft_mode} for epoch {epoch}')
                loss_train = train_loop(
                    dataloader['train'],
                    model,
                    tokenizer,
                    optimizer, scheduler,
                    model_name, is_gpt, is_act, src_len, tgt_len,
                    accelerator, mode=sft_mode)
                
                # 验证
                if hasattr(args, 'do_valid') and args.do_valid:
                    loss_valid, score_df = valid_loop(args,
                        dataloader['valid'], model,
                        tokenizer,
                        graph_samplers,
                        is_gpt, is_act, src_len, tgt_len,
                        verbose)
                else:
                    loss_valid = float('inf')
                    score_df = None
                
                loss_log[f'train_{sft_mode}'][epoch] = loss_train
                loss_log[f'valid_{sft_mode}'][epoch] = loss_valid
                handle_epoch_end(epoch, loss_train, loss_valid, score_df, sft_mode, result_path, loss_log, args, model, optimizer, scheduler, model_name, nepoch)
            
            # scheduler步进（每个epoch结束后）
            if ('-5' in model_name or '-01' in model_name):
                scheduler.step()




def test_loop(args, dataloader, model, tokenizer, graph_samplers, searching_split, resume_epoch,
            is_gpt, is_act, src_len, tgt_len,
            accelerator,
            score_file_suffix='test'):
    score_file_suffix = f'test|{args.test_proportion}x{args.test_split}_topk{args.test_top_k}_{args.constrained}_{args.test_count0}'
    
    # print(len(dataloader))
    if not accelerator is None:
        model, dataloader = accelerator.prepare(
            model, dataloader
        )
    # print(len(dataloader))
    model.eval()
    print(model.device)
    niter = len(dataloader)
    # total_loss = 0

    # initialization
    # scores_stat = initialize_scores_stat(pattern_filtered)
    scores_all = []
    pattern_id_all = []
    failures = []
    
    # 添加training_step结果的统计（仅在需要对比时）
    if args.compare_training_step:
        training_scores_all = []
        training_pattern_id_all = []

    # print('# tgt_len', tgt_len)

    import torch.distributed as dist
    with torch.no_grad():
        for iter, sample in (pbar := tqdm(enumerate(dataloader, start=1),
                                          total=niter, disable=(accelerator is not None) and (not accelerator.is_local_main_process))):
            # gathered_sample = accelerator.gather_for_metrics(sample) if accelerator is not None else sample
            
           
            # 根据开关决定使用哪种方法
            if args.compare_training_step:
                # 使用training_step获取模型输出（不采样）
                source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
                new_extract_sample_to_device(device, sample, tokenizer, is_gpt, src_len, tgt_len, False)
                loss, logits = model.training_step(inputs=input_ids, source_attention_mask=source_attention_mask)
                training_pred = logits.argmax(2)
                
                if is_gpt:
                    mask_source(device, source_attention_mask, training_pred, tokenizer)
                
                training_pred_decoded = tokenizer.batch_decode(training_pred, skip_special_tokens=True)

                # 打印training_step结果
                print('=' * 50)
                print(f'Batch {iter}:')
                print('target (label):')
                print(target[:3])
                print('training_step prediction:')
                print(training_pred_decoded[:3])
                print('=' * 50)
            else:
                # 采样结果
                source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_attention_mask = \
                unify_extract_sample_to_device(device, sample, tokenizer, is_gpt, src_len, tgt_len, True)
                
                # pred, pred_list = model.diff_sample_unify(prompt=input_ids, 
                #     context_length=src_len+tgt_len, device=device, source_attention=source_attention_mask)
                pred = model.diffusion_generate(inputs=input_ids, attention_mask=source_attention_mask)
                
                # print(input_ids.shape)
                # print('pred')
                # print(pred[:10])

                if is_gpt: 
                    mask_source(device, source_attention_mask, pred, tokenizer)
                
                pred_decoded = tokenizer.batch_decode(pred, skip_special_tokens=True)
                
                # 正常显示采样结果
                print('target (label):')
                print(target[:3])
                print('sampling prediction:')
                print(pred_decoded[:3])
            
            scoring_fn = scoring_input_act_batch if is_act else scoring_input_wordlist_batch
            
            if args.compare_training_step:
                # 评估training_step结果
                scores, failures_batch_id = scoring_fn(
                    pred_word_batch=training_pred_decoded,
                    label_word_batch=target,
                    ans_word_batch=source,
                    scoring_method=['smatch', 'precrecf1', 'jaccard'] + ['count0'] * (args.test_count0 == True),
                    do_correction=args.do_correction,
                    graph_samplers=graph_samplers,
                    searching_split=searching_split,
                    return_failures=True,
                    verbose=args.vs)
                
                print(f'Training scores: {scores[:3]}')
            else:
                # 评估采样结果
                scores, failures_batch_id = scoring_fn(
                    pred_word_batch=pred_decoded,
                    label_word_batch=target,
                    ans_word_batch=source,
                    scoring_method=['smatch', 'precrecf1', 'jaccard'] + ['count0'] * (args.test_count0 == True),
                    do_correction=args.do_correction,
                    graph_samplers=graph_samplers,
                    searching_split=searching_split,
                    return_failures=True,
                    verbose=args.vs)
                
                print(f'Sampling scores: {scores[:3]}')
            
            if accelerator is not None:
                gathered_scores = [None] * accelerator.num_processes
                dist.all_gather_object(gathered_scores, scores)
                gathered_scores = [s for l in gathered_scores for s in l]
                gathered_pattern_id = accelerator.gather(pattern_id)
            else:
                gathered_scores = scores
                gathered_pattern_id = pattern_id

            if (accelerator is None) or (accelerator.is_main_process):
                scores_all.extend(gathered_scores)
                pattern_id_all.extend(gathered_pattern_id)
                
                score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
                
                if args.compare_training_step:
                    # 显示training_step统计
                    training_smatch = round(score_df.loc["all",("smatch","mean")], 4)
                    training_jaccard = round(score_df.loc["all",("jaccard","mean")], 4)
                    pbar.set_description(f'Training - s: {training_smatch}, j: {training_jaccard}')
                    
                    # 保存training_step结果
                    training_scores_path = os.path.join(args.result_root, args.modelname,\
                        f'{args.dataname}-{args.scale}-{args.max_answer_size}-{resume_epoch}-training_scores({score_file_suffix}).csv')
                    score_df.to_csv(training_scores_path)
                else:
                    # 正常显示采样统计
                    sampling_smatch = round(score_df.loc["all",("smatch","mean")], 4)
                    sampling_jaccard = round(score_df.loc["all",("jaccard","mean")], 4)
                    pbar.set_description(f's: {sampling_smatch}, j: {sampling_jaccard}')
                
                    # 保存采样结果
                    scores_path = os.path.join(args.result_root, args.modelname,\
                        f'{args.dataname}-{args.scale}-{args.max_answer_size}-{resume_epoch}-scores({score_file_suffix}).csv')
                    score_df.to_csv(scores_path)

    return score_df

def test_loop_deductive(args, dataloader, model, tokenizer, graph_samplers, searching_split, resume_epoch,
            is_gpt, is_act, src_len, tgt_len,
            accelerator,
            score_file_suffix='test'):
    score_file_suffix = f'test|{args.test_proportion}x{args.test_split}_topk{args.test_top_k}_{args.constrained}_{args.test_count0}'
    
    # print(len(dataloader))
    if not accelerator is None:
        model, dataloader = accelerator.prepare(
            model, dataloader
        )
    # print(len(dataloader))
    model.eval()
    print(model.device)
    niter = len(dataloader)
    # total_loss = 0

    # initialization
    # scores_stat = initialize_scores_stat(pattern_filtered)
    scores_all = []
    pattern_id_all = []
    failures = []
    
    # 添加training_step结果的统计（仅在需要对比时）
    if args.compare_training_step:
        training_scores_all = []
        training_pattern_id_all = []

    # print('# tgt_len', tgt_len)

    import torch.distributed as dist
    with torch.no_grad():
        for iter, sample in (pbar := tqdm(enumerate(dataloader, start=1),
                                          total=niter, disable=(accelerator is not None) and (not accelerator.is_local_main_process))):
            # gathered_sample = accelerator.gather_for_metrics(sample) if accelerator is not None else sample
            
           
            # 根据开关决定使用哪种方法
            if args.compare_training_step:
                print('do not complete this part')
                return 0
            else:
                # 采样结果
                source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_attention_mask = \
                unify_extract_sample_to_device(device, sample, tokenizer, is_gpt, src_len, tgt_len, True)
                print(source[:3])
               
                pred_dict = model.diffusion_generate(inputs=input_ids, attention_mask=target_attention_mask)
                pred = pred_dict['sequences']
                histories = pred_dict['history']
                torch.save(histories, 'histories.pt')
                # print(input_ids.shape)
                print('pred')
                print(pred[:3])

                if is_gpt: 
                    mask_source(device, target_attention_mask, pred, tokenizer)
                
                pred_decoded = tokenizer.batch_decode(pred, skip_special_tokens=True)
                
                # 正常显示采样结果
                print('target (label):')
                print(source[:3])
                print('sampling prediction:')
                print(pred_decoded[:3])
            
            # scoring_fn = scoring_input_act_batch if is_act else scoring_input_wordlist_batch
            
            # if args.compare_training_step:
            #     # 评估training_step结果
            #     scores, failures_batch_id = scoring_fn(
            #         pred_word_batch=training_pred_decoded,
            #         label_word_batch=target,
            #         ans_word_batch=source,
            #         scoring_method=['smatch', 'precrecf1', 'jaccard'] + ['count0'] * (args.test_count0 == True),
            #         do_correction=args.do_correction,
            #         graph_samplers=graph_samplers,
            #         searching_split=searching_split,
            #         return_failures=True,
            #         verbose=args.vs)
                
            #     print(f'Training scores: {scores[:3]}')
            # else:
            #     # 评估采样结果
            #     scores, failures_batch_id = scoring_fn(
            #         pred_word_batch=pred_decoded,
            #         label_word_batch=target,
            #         ans_word_batch=source,
            #         scoring_method=['smatch', 'precrecf1', 'jaccard'] + ['count0'] * (args.test_count0 == True),
            #         do_correction=args.do_correction,
            #         graph_samplers=graph_samplers,
            #         searching_split=searching_split,
            #         return_failures=True,
            #         verbose=args.vs)
                
            #     print(f'Sampling scores: {scores[:3]}')
            
            # if accelerator is not None:
            #     gathered_scores = [None] * accelerator.num_processes
            #     dist.all_gather_object(gathered_scores, scores)
            #     gathered_scores = [s for l in gathered_scores for s in l]
            #     gathered_pattern_id = accelerator.gather(pattern_id)
            # else:
            #     gathered_scores = scores
            #     gathered_pattern_id = pattern_id

            # if (accelerator is None) or (accelerator.is_main_process):
            #     scores_all.extend(gathered_scores)
            #     pattern_id_all.extend(gathered_pattern_id)
                
            #     score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
                
            #     if args.compare_training_step:
            #         # 显示training_step统计
            #         training_smatch = round(score_df.loc["all",("smatch","mean")], 4)
            #         training_jaccard = round(score_df.loc["all",("jaccard","mean")], 4)
            #         pbar.set_description(f'Training - s: {training_smatch}, j: {training_jaccard}')
                    
            #         # 保存training_step结果
            #         training_scores_path = os.path.join(args.result_root, args.modelname,\
            #             f'{args.dataname}-{args.scale}-{args.max_answer_size}-{resume_epoch}-training_scores({score_file_suffix}).csv')
            #         score_df.to_csv(training_scores_path)
            #     else:
            #         # 正常显示采样统计
            #         sampling_smatch = round(score_df.loc["all",("smatch","mean")], 4)
            #         sampling_jaccard = round(score_df.loc["all",("jaccard","mean")], 4)
            #         pbar.set_description(f's: {sampling_smatch}, j: {sampling_jaccard}')
                
            #         # 保存采样结果
            #         scores_path = os.path.join(args.result_root, args.modelname,\
            #             f'{args.dataname}-{args.scale}-{args.max_answer_size}-{resume_epoch}-scores({score_file_suffix}).csv')
            #         score_df.to_csv(scores_path)

    return score_df

from trl import (AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead)
from akgr.abduction_model.myconfig import GRPOConfig
from akgr.abduction_model.coupled_mygrpo import DiffuGRPOTrainer
def reward_add(score:dict):
    
    return score['jaccard'] * rl_factor[0] + score['smatch'] * rl_factor[1] 
    

def reward_func(completions, target, source, **kwargs):

    scores, failures_batch_id = scoring_input_act_batch(
            pred_word_batch=completions,
            label_word_batch=target,
            ans_word_batch=source,
            scoring_method=rl_scoring_list,
            do_correction=do_correction,
            graph_samplers=graph_samplers,
            searching_split=rl_search_split,
            return_failures=True,
            )
    return [torch.tensor(reward_add(score), dtype=torch.float) for score in scores]


def optimize_gpro(args, dataset, model, tokenizer, graph_sampler, batch_size,
             is_gpt, is_act, src_len, tgt_len):
    print('GRPO Setting Up')

    # Prepare dataset
    dataset_path = os.path.join(
        args.data_root, args.dataname,
        f'{args.dataname}-{args.scale}-{args.max_answer_size}-train-unify-{args.rl_proportion}')
    if os.path.exists(dataset_path):
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
    else:
        from functools import partial
        from akgr.tokenizer import source_to_prompt
        bound_func = partial(
        source_to_prompt,
        args = args
        )
        dataset = dataset.map(bound_func)
        dataset.set_format(type="torch")
        dataset.save_to_disk(dataset_path)
   
# prepare config
    grpo_config = GRPOConfig(
        seed=42, # default
        output_dir=f'results2/optim/{args.dataname}',
        num_train_epochs=args.rl_epochs,
        learning_rate=args.rl_lr,
        beta=args.rl_init_kl_coef, #do not knows
        epsilon=args.rl_cliprange,
        num_generations=4,
        num_iterations=args.rl_num_iterations,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size = batch_size,
        remove_unused_columns=False, # Important. By default, it removes unrecognized columns if hf dataset is passed
        report_to='tensorboard',
        # 保存相关参数
        save_strategy="steps",  # 按步数保存
        save_steps=args.rl_save_steps,  # 每N步保存一次
        save_total_limit=args.rl_save_total_limit,  # 最多保存N个检查点
        )
    # print(grpo_config)
    
    global graph_samplers
    graph_samplers = graph_sampler
    global rl_search_split
    rl_search_split = args.rl_search_split
    global do_correction
    do_correction = args.do_correction
    global rl_scoring_list
    rl_scoring_list=['jaccard','smatch'] 
    global rl_factor
    print(eval(args.rl_factor))
    rl_factor = eval(args.rl_factor)

    model.warnings_issued = {}
    def dummy_add_model_tags(self, tags):
        pass

    model.add_model_tags = dummy_add_model_tags.__get__(model)

    trainer = DiffuGRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        processing_class=tokenizer
        )
    # trainer.train()
    # ckpt_path = os.path.join(args.checkpoint_root, args.modelname,\
    #             f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.rl_epochs}-optimize.pth')
    # save_model(ckpt_path, 'model', model, optimizer=None, scheduler=None, epoch=args.rl_epochs)
    return 0

def load_model_by_mode(args, device, model_name, is_gpt, config_model=None, ntoken=None, config_train=None):
    # 总是初始化一个新的 Diffusion 对象
    print('Creating model')
    model = Diffusion(ntoken=ntoken,
        special_tokens=special_tokens,
        model_name=model_name,
        config_model=config_model,
        device=device,
        drop=args.drop,
        generation_config=args.generation_config
        )
    print(model)

    if args.mode in ['training', 'testing'] and args.resume_epoch != 0 and args.training_mode == 'unify':
        # 加载保存的模型参数
        resume_path = os.path.join(args.checkpoint_root, args.modelname,\
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.resume_epoch}.pth')
        print(f'Loading model parameters: {resume_path}')
        loaded_model, optimizer, scheduler, last_epoch, loss_log = \
            load_model(resume_path, 'model', args.resume_epoch, return_huggingface_model=False)
        loaded_model.to(device)
        
        # 用加载的模型参数替换 Diffusion 中的模型
        model.model = loaded_model.model
        
        last_epoch = 0
    elif args.mode in ['training', 'optimizing'] and args.resume_epoch != 0 and args.training_mode == 'sft':
        # 加载保存的模型参数
        resume_path = os.path.join(args.checkpoint_root, args.modelname,\
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.resume_epoch}.pth')
        print(f'Loading model parameters: {resume_path}')
        loaded_model, optimizer, scheduler, last_epoch, loss_log = \
            load_model(resume_path, 'model', args.resume_epoch, return_huggingface_model=False)
        loaded_model.to(device)
        model.model = loaded_model.model
        # 初始化优化器和调度器
        optimizer = torch.optim.Adam(model.parameters(),
            lr=float(config_train["lr"]))
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
            start_factor=0.1, total_iters=config_train["warm_up"])
        last_epoch = 0
        loss_log = {'train': {}, 'valid': {}}

    elif args.mode in ['optimizing'] and args.rl_resume_epoch == 0 :
        # 加载保存的模型参数
        assert args.resume_epoch != 0
        resume_path = os.path.join(args.checkpoint_root, args.modelname,\
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-sft_target-{args.resume_epoch}.pth')
        print(f'Loading model parameters: {resume_path}')
        loaded_model, optimizer, scheduler, last_epoch, loss_log = \
            load_model(resume_path, 'model', args.resume_epoch, return_huggingface_model=False)
        loaded_model.to(device)
        model.model = loaded_model.model
        model.to(device)

    elif args.mode == 'testing' and args.resume_epoch != 0 and args.training_mode == 'sft':
        # 加载保存的模型参数
        resume_path = os.path.join(args.checkpoint_root, args.modelname,\
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-sft_source-{args.resume_epoch}.pth')
        print(f'Loading model parameters: {resume_path}')
        loaded_model, optimizer, scheduler, last_epoch, loss_log = \
            load_model(resume_path, 'model', args.resume_epoch, return_huggingface_model=False)
        loaded_model.to(device)
        model.model = loaded_model.model
        model.to(device)
        last_epoch = 0
        loss_log = {'train': {}, 'valid': {}}

    elif args.mode == 'testing' and args.rl_resume_epoch != 0 :
        resume_path = '/home/zhangziwei/gys/AbductiveKGR/results2/optim/DBpedia50/checkpoint-177000'
        print(f'Loading model parameters: {resume_path}')
        grpo_state_dict, _ = load_grpo_model(resume_path, device)
        
        # 现在grpo_state_dict直接是state_dict，但需要提取主模型部分并去掉前缀
        print(f"✅ 直接获得GPT2格式的state_dict，包含{len(grpo_state_dict)}个参数")
        
        # 提取并重新映射所有模型权重
        mapped_state_dict = {}
        
        for key, value in grpo_state_dict.items():
            if key.startswith('model.model.'):
                # 去掉'model.'前缀: model.model.transformer.* -> model.transformer.*
                new_key = key[6:]
                mapped_state_dict[new_key] = value
            elif key.startswith('model.embed_tokens.'):
                # 直接映射: model.embed_tokens.weight -> embed_tokens.weight
                new_key = key[6:]
                mapped_state_dict[new_key] = value
            elif key.startswith('model.denoise_model.'):
                # 直接映射: model.denoise_model.* -> denoise_model.*
                new_key = key[6:]
                mapped_state_dict[new_key] = value
            elif key.startswith('model.lm_head.'):
                # 直接映射: model.lm_head.weight -> lm_head.weight
                new_key = key[6:]
                mapped_state_dict[new_key] = value
        
        print(f"\n=== 重新映射所有权重 ===")
        print(f"映射后参数数量: {len(mapped_state_dict)}")
        
        # 显示映射统计
        mapping_stats = {}
        for key in mapped_state_dict.keys():
            if '.' in key:
                prefix = key.split('.')[0]
                mapping_stats[prefix] = mapping_stats.get(prefix, 0) + 1
        
        print("映射统计:")
        for prefix, count in mapping_stats.items():
            print(f"  {prefix}: {count} 参数")
        
        # 加载权重到model.model
        missing_keys, unexpected_keys = model.model.load_state_dict(mapped_state_dict, strict=False)
        print(f"\n=== 加载结果 ===")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        # 显示匹配情况
        total_params = len(model.model.state_dict())
        loaded_params = total_params - len(missing_keys)
        print(f"✅ 成功加载: {loaded_params}/{total_params} 参数")
        print(f"✅ 成功率: {loaded_params/total_params*100:.1f}%")
        
        model.to(device)
        last_epoch = 0
        loss_log = {'train': {}, 'valid': {}}
    else:
        optimizer = torch.optim.Adam(model.parameters(),
            lr=float(config_train["lr"]))
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
            start_factor=0.1, total_iters=config_train["warm_up"])
        last_epoch = 0
        loss_log = {'train': {}, 'valid': {}}

    trl_model_class = AutoModelForCausalLMWithValueHead if is_gpt else AutoModelForSeq2SeqLMWithValueHead

    # if args.mode in ['optimizing', 'testing'] and args.rl_resume_epoch != 0: # Load TRL model wrapper directly
    #     resume_dir = os.path.join(args.checkpoint_root, args.modelname,\
    #             f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.rl_resume_epoch}-optimize.pth')
    #     print(f'Loading model: {resume_dir}')
    #     if args.rl_type=='GRPO':
    #         model, optimizer, scheduler, last_epoch, loss_log = \
    #         load_model(resume_path, 'rlmodel', args.resume_epoch, return_huggingface_model=True)
        # model.to(device)

    
    if args.mode == 'training': return model, optimizer, scheduler, last_epoch, loss_log
    else: return model

def my_parse_args():
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument('--modelname')
    parser.add_argument('--config-dataloader', default='akgr/configs/config-dataloader.yml')
    parser.add_argument('--config-train', default='akgr/configs/config-train.yml')
    parser.add_argument('--config-model', default='akgr/configs/config-model.yml')
    parser.add_argument('--config-batchsize', default='akgr/configs/config-batchsize.yml')
    parser.add_argument('--config-diffusion', default='akgr/configs/config-diffusion.yml')
    parser.add_argument('--overwrite_batchsize', type=int, default=0)

    # Data
    parser.add_argument('--data_root', default='./sampling/')
    parser.add_argument('-d', '--dataname', default='FB15k-237')
    parser.add_argument('--scale', default='debug')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)

    # Checkpoint
    parser.add_argument('--checkpoint_root', default='./checkpoints/')
    parser.add_argument('-r', '--resume_epoch', type=int, default=0)

    parser.add_argument('--vs', action='store_true', help='verbose flag for smatch result')
    parser.add_argument('--do_correction', action='store_true', help='verbose flag for smatch result')

    parser.add_argument('--drop', type=float, default=0.)

    #training
    parser.add_argument('--training_mode', default='unify')

    #valid
    parser.add_argument('--do_valid', action='store_true', help='verbose flag for smatch result')
    parser.add_argument('--do_initial_valid', action='store_true', help='verbose flag for smatch result')

    # Testing
    parser.add_argument('--test_proportion', type=float, default=1)
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--test_top_k', type=int, default=0)
    parser.add_argument('--test_count0', action='store_true')
    parser.add_argument('--result_root', default='./results/')

    parser.add_argument('--save_frequency', type=int, default=1)

    # rl
    parser.add_argument('--rl_type', default='GRPO')
    parser.add_argument('--rl_resume_epoch', default=0)
    parser.add_argument('--rl_smatch_factor', type=float, default=0)
    parser.add_argument('--rl_init_kl_coef', type=float, default=0.2)
    parser.add_argument('--rl_cliprange', type=float, default=0.2)
    parser.add_argument('--rl_minibatch', type=int, default=1)
    parser.add_argument('--rl_horizon', type=int, default=10000)
    parser.add_argument('--rl_lr', type=float)
    parser.add_argument('--rl_epochs', type=int, default=4)
    parser.add_argument('--rl_search_split', default='train')
    parser.add_argument('--rl_share_embed_layer', action='store_true')
    parser.add_argument('--rl_lr_no_decay', action='store_true')
    parser.add_argument('--rl_use_peft', action='store_true')
    parser.add_argument('--rl_top_k', default=0.0)
    parser.add_argument('--rl_factor', type=str, default='[1.0, 2.0]',)
    parser.add_argument('--rl_proportion', type=float, default=1)
    parser.add_argument('--rl_save_steps', type=int, default=3000, help='Save checkpoint every N steps')
    parser.add_argument('--rl_save_total_limit', type=int, default=5, help='Maximum number of checkpoints to keep')
    parser.add_argument('--rl_num_iterations', type=int, default=2)

    parser.add_argument('--mode')
    parser.add_argument('--accelerate', action='store_true')
    parser.add_argument('--constrained', action='store_true')
    parser.add_argument('--compare_training_step', action='store_true', help='Compare training_step results with sampling results')

    # parser.add_argument('--wandb_run_id', default=None)

    args = parser.parse_args()
    return args

def main():
    args = my_parse_args()
    print(f'# Running main.py in {args.mode} mode with:')
    print(f'args:\n{args}\n')

    if not os.path.exists(os.path.join(args.result_root, args.modelname)):
        os.makedirs(os.path.join(args.result_root, args.modelname))

    # Data representation
    global config_dataloader
    config_dataloader = load_yaml(args.config_dataloader)
    global offset, special_tokens
    offset = config_dataloader['offset']
    special_tokens = config_dataloader['special_tokens']
    print(f'config_dataloader:\n{config_dataloader}\n')

    # Generation configuration
    config_diffusion = load_yaml(args.config_diffusion)
    
    # 将字典转换为对象属性
    class ConfigObject:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                # 处理字符串'None'转换为真正的None
                if value == 'None':
                    value = None
                setattr(self, key, value)
    
    args.generation_config = ConfigObject(config_diffusion)

    global pattern_filtered
    pattern_filtered_path = 'akgr/metadata/pattern_filtered.csv'
    pattern_filtered = pd.read_csv(pattern_filtered_path, index_col='id')

    # Graphs (for evaluation)
    print('Loading graph')
    kg = load_kg(args.dataname)
    graph_samplers = kg.graph_samplers

    # Device
    global device
    if args.accelerate and args.mode != 'optimizing':
        accelerator = Accelerator()
        device = accelerator.device
    else:
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    # Model information
    model_name = args.modelname
    is_gpt=('GPT2' in model_name)
    is_act=('act' in model_name)
    tgt_len = config_dataloader['act_len'] + 1 if is_act else config_dataloader['qry_len'] + 1
    src_len = config_dataloader['ans_len'] + 1
    print(f'model_name:{model_name}\n')

    # Batch size
    config_batchsize = load_yaml(args.config_batchsize)
    batch_size = config_batchsize[model_name][args.dataname]
    if args.overwrite_batchsize != 0:
        batch_size = args.overwrite_batchsize
    print(f'batch_size:{batch_size}\n')

    print('=' * 50)

    # Dataset
    if args.mode == 'training':  splits = ['train', 'valid']
    elif args.mode == 'testing': splits = [args.test_split]
    elif args.mode == 'optimizing':
        splits = ['train']
    elif args.mode == 'load-save-test': splits = ['train', 'test']

    print('Creating dataset & dataloader')
    global nentity, nrelation
    dataset_dict, nentity, nrelation = new_create_dataset(
        dataname=args.dataname,
        scale=args.scale,
        answer_size=args.max_answer_size,
        pattern_filtered=pattern_filtered,
        data_root=args.data_root,
        splits=splits,
        is_act=is_act
    )
    if args.mode == 'testing' and args.test_proportion < 1:
        nrows = dataset_dict[args.test_split].shape[0]
        dataset_dict[args.test_split] = dataset_dict[args.test_split].select(random.sample(range(nrows), int(nrows * args.test_proportion)))
    if args.mode == 'optimizing' and args.rl_proportion < 1:
        nrows = dataset_dict['train'].shape[0]
        dataset_dict['train'] = dataset_dict['train'].select(random.sample(range(nrows), int(nrows * args.rl_proportion)))
    dataloader_dict = new_create_dataloader(
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        drop_last=(args.mode == 'optimizing') #or (args.mode == 'testing' and args.accelerate)
    )
    
    # Tokenizer
    print('Creating tokenizer')
    tokenizer, ntoken = create_tokenizer(
        special_tokens=special_tokens,
        offset=offset,
        nentity=nentity,
        nrelation=nrelation,
           is_gpt=is_gpt
    )
   
    # Model
    config_model = load_yaml(args.config_model)
    config_train = load_yaml(args.config_train)
    if model_name in config_train:
        config_train = config_train[model_name]
    else:
        config_train = config_train['default']
        warnings.warn(f'No training configuration specified for {model_name}')
    print(f'config_train:\n{config_train}')

    if args.mode == 'training':
        model, optimizer, scheduler, last_epoch, loss_log = load_model_by_mode(
            args=args, device=device, model_name=model_name, is_gpt=is_gpt,
            config_model=config_model, ntoken=ntoken, config_train=config_train)
    else:
        model = load_model_by_mode(
            args=args, device=device, model_name=model_name, is_gpt=is_gpt,
            config_model=config_model, ntoken=ntoken, config_train=config_train)
    
    

    if args.mode == 'training':
        # https://huggingface.co/docs/transformers/training#train-in-native-pytorch
        nepoch = config_train['nepoch']
        fit(args, nepoch, dataloader_dict, model,
            tokenizer, optimizer, scheduler, graph_samplers,
            model_name, is_gpt, is_act, src_len, tgt_len,
            last_epoch, loss_log,
            args.vs,
            accelerator=accelerator if args.accelerate else None,
            training_mode=args.training_mode)
    elif args.mode == 'testing':
        # preprocess_allowed_rel_ent_map(graph_samplers)
        result = test_loop_deductive(
            args=args,
            dataloader=dataloader_dict[args.test_split],
            model=model,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            searching_split=args.test_split,
            resume_epoch=args.resume_epoch,
            is_gpt=is_gpt, is_act=is_act,
            src_len=src_len, tgt_len=tgt_len,
            accelerator=accelerator if args.accelerate else None)
        
        score_df = result
        # 打印最终结果
        print('\n' + '=' * 80)
        if args.compare_training_step:
            print('FINAL TRAINING STEP RESULTS:')
            print('=' * 80)
            print(f'Training Step Results:')
        else:
            print('FINAL SAMPLING RESULTS:')
            print('=' * 80)
            print(f'Sampling Results:')
        print(f"  Smatch: {round(score_df.loc['all',('smatch','mean')], 4)}")
        print(f"  Jaccard: {round(score_df.loc['all',('jaccard','mean')], 4)}")
        print(f"  Precision: {round(score_df.loc['all',('precrecf1','precision')], 4)}")
        print(f"  Recall: {round(score_df.loc['all',('precrecf1','recall')], 4)}")
        print(f"  F1: {round(score_df.loc['all',('precrecf1','f1')], 4)}")
        print('=' * 80)
    elif args.mode == 'optimizing':
        if args.rl_type == 'GRPO':
            model = optimize_gpro(
                args=args,
                dataset=dataset_dict['train'],
                model=model,
                tokenizer=tokenizer,
                graph_sampler=graph_samplers,
                batch_size=batch_size,
                is_gpt=is_gpt, is_act=is_act,
                src_len=src_len, tgt_len=tgt_len
            )
        else:
            print('ppo is writing now')

if __name__ == '__main__':
    main()