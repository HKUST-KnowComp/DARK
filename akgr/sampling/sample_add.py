import torch
from akgr.utils.parsing_util import unshift_entity_index, unshift_relation_index, shift_entity_index, shift_relation_index, ans_unshift_indices,\
                                     ans_shift_indices, qry_shift_indices, qry_unshift_indices, qry_actionstr_2_wordlist
from tqdm import tqdm
import os, sys, argparse, warnings

import json
import yaml
import pandas as pd

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import random

# dataloader
from akgr.dataloader import new_create_dataloader, new_create_dataset
from akgr.tokenizer import create_tokenizer, new_extract_sample_to_device

# transformer (huggingface)
from akgr.abduction_model.transformer import create_transformer

# utils
from akgr.utils.stat_util import stat_scores_by_pattern#, initialize_scores_stat
from akgr.utils.load_util import load_yaml, load_model, save_model, load_and_filter_query_patterns
from akgr.kgdata import load_kg
import pandas as pd

# evaluation
from akgr.evaluation import scoring_input_wordlist_batch, scoring_input_act_batch
from akgr.utils.parsing_util import qry_actionprefix_get_branching, is_strint

import wandb
from accelerate import Accelerator
# 全局变量避免重复序列化
global_dataset = None
global_graph = None
def search_answer(target,graph):
    qry_tgt = qry_actionstr_2_wordlist(target)
    
    pred_qry_unshifted = qry_unshift_indices(qry_tgt)
#     # print('unshifted')
    
    pred_ans_unshifted = graph.search_answers_to_query(pred_qry_unshifted)
    ans = ans_shift_indices(pred_ans_unshifted)
    ans_str = ' '.join(map(str, ans))
    return [ans_str]
def p2_to_p(target, graph):
    # s_list = list(map(int, src.split()))
    # src_set = set(ans_unshift_indices(s_list))
    
    p1, p2, e2 = target.split(" ")
    p1 = -1*unshift_relation_index(p1)
    p2 = -1*unshift_relation_index(p2)
    e2 = unshift_entity_index(e2)
    # print(graph.out_edges(e2))
    if e2 not in graph.graph.nodes:
        # print(f"Warning: Node {e2} is not in the graph. Skipping this target.")
        return [], []  # 返回空列表
    waiting_set = {
    v for u, v, k in graph.out_edges(e2) if k == p2 }
    # print(waiting_set)
    new_source_list = []
    new_target_list = []
    if waiting_set:
        for v in waiting_set:
            if v not in graph.graph.nodes:
                continue
            filtered_items = [str(shift_entity_index(w)) for _, w, k in graph.out_edges(v) if k == p1]
            new_source = " ".join(filtered_items)
            if filtered_items:
                new_source_list.append(new_source)
                new_target_list.append( " ".join([str(shift_relation_index(p1)), str(shift_entity_index(v))]))
    
    return new_source_list, new_target_list

def i2_to_p(target,graph):
    # s_list = list(map(int, src.split()))
    i, p1, e1, p2, e2 = target.split(" ")
    target_1p_1 = " ".join([p1,e1])
    source_1p_1 = search_answer(target_1p_1,graph)
    target_1p_2 = " ".join([p2,e2])
    source_1p_2 = search_answer(target_1p_2,graph)
    # p1 = -1*unshift_relation_index(p1)
    # p2 = -1*unshift_relation_index(p2)
    # e1 = unshift_entity_index(e1)
    # e2 = unshift_entity_index(e2)
    # new_source_list = []
    # new_target_list = []
    # filtered_items = [str(shift_entity_index(w)) for _, w, k in graph.out_edges(e1) if k == p1]
    # new_source = " ".join(filtered_items)
    # if filtered_items:
    #     new_source_list.append(new_source)
    #     new_target_list.append( " ".join([str(shift_relation_index(p1)), str(shift_entity_index(e1))]))

    # filtered_items = [str(shift_entity_index(w)) for _, w, k in graph.out_edges(e2) if k == p2]
    # new_source = " ".join(filtered_items)
    # if filtered_items:
    #     new_source_list.append(new_source)
    #     new_target_list.append( " ".join([str(shift_relation_index(p2)), str(shift_entity_index(e2))]))
    merged_src = source_1p_1+source_1p_2
    merged_tgt = [target_1p_1] + [target_1p_2]
    return merged_src, merged_tgt

def i3_to_i2_to_p(target,graph):
    i, i, p1, e1, p2, e2, p3, e3 = target.split(" ")
    target_2i = " ".join([i,p1,e1,p2,e2])
    source_2i = search_answer(target_2i,graph)
    target_1p = " ".join([p3,e3])
    source_1p = search_answer(target_1p,graph)
    # print(target_2i)
    new_src_2i, new_tgt_2i = i2_to_p(target_2i,graph)

    merged_source = source_1p + new_src_2i + source_2i
    merged_target = [target_1p] + new_tgt_2i + [target_2i]
    # need to consider how to caculate

    return merged_source, merged_target

def pi_to_2p_to_p(target,graph):
    i, p, e1, p1, p2, e2 =  target.split(" ")
    target_2p1 = " ".join([p1,p2,e2])
    source_2p1 = search_answer(target_2p1,graph)
    target_2p2 = " ".join([p, e1])
    source_2p2 = search_answer(target_2p2,graph)
   
    new_src_2p1, new_tgt_2p1 = p2_to_p(target_2p1,graph)
    merged_source =  new_src_2p1  + source_2p1 + source_2p2
    merged_target =  new_tgt_2p1  + [target_2p1] + [target_2p2]
    return merged_source, merged_target

def ip_to_2p_to_p(target,graph):
    p, i, p1, e1, p2, e2 =  target.split(" ")
    target_2p1 = " ".join([p,p1,e1])
    source_2p1 = search_answer(target_2p1,graph)
    target_2p2 = " ".join([p,p2,e2])
    source_2p2 = search_answer(target_2p2,graph)
    p1 = -1*unshift_relation_index(p1)
    p2 = -1*unshift_relation_index(p2)
    p = -1*unshift_relation_index(p)
    e1 = unshift_entity_index(e1)
    e2 = unshift_entity_index(e2)
    new_src_2p1, new_tgt_2p1 = p2_to_p(target_2p1,graph)
    new_src_2p2, new_tgt_2p2 = p2_to_p(target_2p2,graph)
    merged_source =  new_src_2p1 + new_src_2p2 + source_2p1 + source_2p2
    merged_target =  new_tgt_2p1 + new_tgt_2p2 + [target_2p1] + [target_2p2]
    return merged_source, merged_target

def u2_to_p(target,graph):
    u, p1, e1, p2, e2 = target.split(" ")
    target_1p_1 = " ".join([p1,e1])
    source_1p_1 = search_answer(target_1p_1,graph)
    target_1p_2 = " ".join([p2,e2])
    source_1p_2 = search_answer(target_1p_2,graph)
    # p1 = -1*unshift_relation_index(p1)
    # p2 = -1*unshift_relation_index(p2)
    # e1 = unshift_entity_index(e1)
    # e2 = unshift_entity_index(e2)
    # new_source_list = []
    # new_target_list = []
    # filtered_items = [str(shift_entity_index(w)) for _, w, k in graph.out_edges(e1) if k == p1]
    # new_source = " ".join(filtered_items)
    # if filtered_items:
    #     new_source_list.append(new_source)
    #     new_target_list.append( " ".join([str(shift_relation_index(p1)), str(shift_entity_index(e1))]))

    # filtered_items = [str(shift_entity_index(w)) for _, w, k in graph.out_edges(e2) if k == p2]
    # new_source = " ".join(filtered_items)
    # if filtered_items:
    #     new_source_list.append(new_source)
    #     new_target_list.append( " ".join([str(shift_relation_index(p2)), str(shift_entity_index(e2))]))
    merged_src = source_1p_1+source_1p_2
    merged_tgt = [target_1p_1] + [target_1p_2]

# #类似2i
    return merged_src, merged_tgt

def in2_to_p(target,graph):
    i, n, p1, e1, p2, e2 = target.split(" ")
    target_1p = " ".join([p2,e2])
    # p2 = unshift_entity_index(e1)
    # e2 = unshift_entity_index(e2)
    # new_source_list = []
    # new_target_list = []
    # filtered_items = [str(shift_entity_index(w)) for _, w, k in graph.out_edges(e1) if k == p1]
    # new_source = " ".join(filtered_items)
    # if filtered_items:
    #     new_source_list.append(new_source)
    #     new_target_list.append( " ".join([str(shift_relation_index(p1)), str(shift_entity_index(e1))]))
    source_1p = search_answer(target_1p,graph)
    return source_1p, [target_1p]

def in3_to_p(target,graph):
    i, i, n, p1, e1, p2, e2, p3, e3 = target.split(" ")
    target_in2_1 = " ".join([i, n, p1, e1, p2,e2])
    source_in2_1 = search_answer(target_in2_1,graph)
    target_in2_2 = " ".join([i, n, p1, e1, p3,e3])
    source_in2_2 = search_answer(target_in2_2,graph)
    target_i2 = " ".join([i, p2, e2, p3, e3])
    source_i2 = search_answer(target_i2, graph)
    new_src_2p1, new_tgt_2p1 = in2_to_p(target_in2_1,graph)
    new_src_2p2, new_tgt_2p2 = in2_to_p(target_in2_2,graph)
    merged_source =  new_src_2p1 + new_src_2p2 + source_in2_1 + source_in2_2 + source_i2
    merged_target =  new_tgt_2p1 + new_tgt_2p2 + [target_in2_1] + [target_in2_2] + [target_i2]
#构造2i，2in，1p
    return merged_source, merged_target

def inp_to_p(target,graph):
    p, i, n, p1, e1, p2, e2 = target.split(" ")
    target_2p = " ".join([p, p2, e2])
    source_2p = search_answer(target_2p,graph)
    new_src_2p1, new_tgt_2p1 = p2_to_p(target_2p,graph)
    merged_source =  new_src_2p1 + source_2p
    merged_target =  new_tgt_2p1 + [target_2p]
#构造2p,1p
    return merged_source, merged_target

   

def pni_to_p(target,graph):
    
    i, n, p1, p2, e1, p, e = target.split(" ")
    target_1p = " ".join([p, e])
    source_1p = search_answer(target_1p,graph)
#构造1p
    return source_1p, [target_1p]

def pin_to_p(target,graph):
    i, n, p, e, p1, p2, e1 = target.split(" ")
    target_2p = " ".join([p1, p2, e1])
    source_2p = search_answer(target_2p,graph)
    merged_source =  source_2p
    merged_target =  [target_2p]
    # new_src_2p1, new_tgt_2p1 = p2_to_p(target_2p,graph)
    # merged_source =  new_src_2p1 + source_2p
    # merged_target =  new_tgt_2p1 + [target_2p]
#构造2p,1p
    return merged_source, merged_target

def up_to_p(target,graph):
    #构造2p,1p
    p, u, p1, e1, p2, e2 =  target.split(" ")
    target_2p1 = " ".join([p,p1,e1])
    source_2p1 = search_answer(target_2p1,graph)
    target_2p2 = " ".join([p,p2,e2])
    source_2p2 = search_answer(target_2p2,graph)
    merged_source =  source_2p1 + source_2p2
    merged_target =  [target_2p1] + [target_2p2]

    # p1 = -1*unshift_relation_index(p1)
    # p2 = -1*unshift_relation_index(p2)
    # p = -1*unshift_relation_index(p)
    # e1 = unshift_entity_index(e1)
    # e2 = unshift_entity_index(e2)
    # new_src_2p1, new_tgt_2p1 = p2_to_p(target_2p1,graph)
    # new_src_2p2, new_tgt_2p2 = p2_to_p(target_2p2,graph)
    # merged_source =  new_src_2p1 + new_src_2p2 + source_2p1 + source_2p2
    # merged_target =  new_tgt_2p1 + new_tgt_2p2 + [target_2p1] + [target_2p2]
    return merged_source, merged_target
def check(src, tgt):
    # 过滤掉 src 中为空字符串的项
    filtered_pairs = [(s, t) for s, t in zip(src, tgt) if s != '']
    
    # 如果过滤后的结果为空，返回两个空列表
    if not filtered_pairs:
        return [], []
    
    # 解包并转换为列表
    filtered_src, filtered_tgt = zip(*filtered_pairs)
    filtered_src = list(filtered_src)
    filtered_tgt = list(filtered_tgt)
    
    return filtered_src, filtered_tgt

def replace_numbers(sequence):
        return " ".join("e" if s.lstrip("-").isdigit() and int(s) > 0 else "p" if s.lstrip("-").isdigit() else s for s in sequence.split())
# dataloader_dict = torch.load('dataloader.pt')
# graph_samplers = torch.load('graph_samplers.pt')
# dataloader = dataloader_dict['train']
# print(graph_samplers)
# graph = graph_samplers['train']
# print(graph)
# dataset = torch.load('dataset.pt')
# print(dataset['train'])

def sample_add(dataset,graph,allow_pattern_dict,device_name):
    new_target = []
    new_source = []
    all_targets = dataset['train']['target']
    all_pattern_ids = dataset['train']['pattern_id']
    start_idx = device_name*500000
    end_idx = (device_name+1)*500000
    if end_idx is None or end_idx > len(all_targets):
        end_idx = len(all_targets)
    for i, tgt in tqdm(enumerate(all_targets[start_idx:end_idx]), 
                       total=end_idx - start_idx,
                       desc=f"Processing {start_idx}-{end_idx}"):
        actual_idx = start_idx + i  # 计算原始数据集中的真实索引
        pattern_id = all_pattern_ids[actual_idx]
        if pattern_id == 1 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = p2_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 3 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = ip_to_2p_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 4 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = inp_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 5 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = up_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 6 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = i2_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 7 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = pi_to_2p_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 8 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = in2_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 9 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = pni_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 10 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = pin_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 11 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = u2_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 12 and pattern_id in allow_pattern_dict: 
            new_src, new_tgt = i3_to_i2_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
        elif pattern_id == 13 and pattern_id in allow_pattern_dict:
            new_src, new_tgt = in3_to_p(tgt,graph)
            new_src, new_tgt = check(new_src, new_tgt)
            new_source = new_source + new_src
            new_target = new_target + new_tgt
    return new_source,new_target


# from tqdm import tqdm
# from multiprocessing import Pool
# from functools import partial

# def process_item(item, graph, allow_pattern_dict,dataset):
#     i, tgt = item
#     pattern_id = dataset['train']['pattern_id'][i]
#     new_src, new_tgt = [], []
    
#     if pattern_id == 1 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = p2_to_p(tgt, graph)
#     elif pattern_id == 3 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = ip_to_2p_to_p(tgt, graph)
#     elif pattern_id == 4 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = inp_to_p(tgt, graph)
#     elif pattern_id == 5 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = up_to_p(tgt, graph)
#     elif pattern_id == 6 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = i2_to_p(tgt, graph)
#     elif pattern_id == 7 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = pi_to_2p_to_p(tgt, graph)
#     elif pattern_id == 8 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = in2_to_p(tgt, graph)
#     elif pattern_id == 9 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = pni_to_p(tgt, graph)
#     elif pattern_id == 10 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = pin_to_p(tgt, graph)
#     elif pattern_id == 11 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = u2_to_p(tgt, graph)
#     elif pattern_id == 12 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = i3_to_i2_to_p(tgt, graph)
#     elif pattern_id == 13 and pattern_id in allow_pattern_dict:
#         new_src, new_tgt = in3_to_p(tgt, graph)
    
#     if new_src and new_tgt:
#         new_src, new_tgt = check(new_src, new_tgt)
#         return new_src, new_tgt
#     return [], []

# def sample_add_parallel(dataset, graph, allow_pattern_dict, num_processes=8):
#     items = list(enumerate(dataset['train']['target']))
    
#     # 使用进程池并行处理
#     with Pool(processes=num_processes) as pool:
#         # 使用 partial 固定 graph 和 allow_pattern_dict 参数
#         process_func = partial(process_item, graph=graph, allow_pattern_dict=allow_pattern_dict, dataset=dataset)
        
#         # 使用 tqdm 显示进度条
#         results = list(tqdm(
#             pool.imap(process_func, items),
#             total=len(items),
#             desc="Processing in parallel"
#         ))
    
#     # 合并所有结果
#     new_source = []
#     new_target = []
#     for src, tgt in results:
#         new_source.extend(src)
#         new_target.extend(tgt)
    
#     return new_source, new_target

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_item(item, allow_pattern_dict):
    i, tgt = item
    pattern_id = global_dataset['train']['pattern_id'][i]
    new_src, new_tgt = [], []
    graph = global_graph
    if pattern_id == 1 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = p2_to_p(tgt, graph)
    elif pattern_id == 3 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = ip_to_2p_to_p(tgt, graph)
    elif pattern_id == 4 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = inp_to_p(tgt, graph)
    elif pattern_id == 5 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = up_to_p(tgt, graph)
    elif pattern_id == 6 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = i2_to_p(tgt, graph)
    elif pattern_id == 7 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = pi_to_2p_to_p(tgt, graph)
    elif pattern_id == 8 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = in2_to_p(tgt, graph)
    elif pattern_id == 9 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = pni_to_p(tgt, graph)
    elif pattern_id == 10 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = pin_to_p(tgt, graph)
    elif pattern_id == 11 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = u2_to_p(tgt, graph)
    elif pattern_id == 12 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = i3_to_i2_to_p(tgt, graph)
    elif pattern_id == 13 and pattern_id in allow_pattern_dict:
        new_src, new_tgt = in3_to_p(tgt, graph)
    if new_src and new_tgt:
        new_src, new_tgt = check(new_src, new_tgt)
    return new_src, new_tgt

def sample_add_threaded(dataset, graph, allow_pattern_dict, max_workers=8, batch_size=1000):
    global global_dataset, global_graph
    global_dataset, global_graph = dataset, graph  # 设置全局变量
    
    items = list(enumerate(dataset['train']['target']))
    new_source, new_target = [], []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 分批提交任务
        futures = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            futures.extend(executor.submit(process_item, item, allow_pattern_dict) for item in batch)
            print(f"已提交 {len(futures)} 个任务")  # 打印进度
            
        # 获取结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理中"):
            src, tgt = future.result()
            new_source.extend(src)
            new_target.extend(tgt)
    
    return new_source, new_target
def my_parse_args():
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument('--modelname')
    parser.add_argument('--config-dataloader', default='akgr/configs/config-dataloader.yml')
    parser.add_argument('--config-train', default='akgr/configs/config-train.yml')
    parser.add_argument('--config-model', default='akgr/configs/config-model.yml')
    parser.add_argument('--config-batchsize', default='akgr/configs/config-batchsize.yml')
    parser.add_argument('--overwrite_batchsize', type=int, default=0)

    # Data
    parser.add_argument('--data_root', default='./sampling/')
    parser.add_argument('-d', '--dataname', default='FB15k-237')
    parser.add_argument('--scale', default='debug')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)

    # Checkpoint
    parser.add_argument('--checkpoint_root', default='./ckpt/')
    parser.add_argument('-r', '--resume_epoch', type=int, default=0)

    parser.add_argument('--vs', action='store_true', help='verbose flag for smatch result')
    parser.add_argument('--do_correction', action='store_true', help='verbose flag for smatch result')

    # Testing
    parser.add_argument('--test_proportion', type=float, default=1)
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--test_top_k', type=int, default=0)
    parser.add_argument('--test_count0', action='store_true')
    parser.add_argument('--result_root', default='./results/')

    parser.add_argument('--save_frequency', type=int, default=1)

    

    parser.add_argument('--mode')
    parser.add_argument('--accelerate', action='store_true')
    parser.add_argument('--constrained', action='store_true')

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

    global pattern_filtered
    pattern_filtered_path = 'akgr/metadata/pattern_filtered.csv'
    pattern_filtered = pd.read_csv(pattern_filtered_path, index_col='id')

    # Graphs (for evaluation)
    print('Loading graph')
    kg = load_kg(args.dataname)
    graph_samplers = kg.graph_samplers


    # Model information
    model_name = args.modelname
    
    is_act=('act' in model_name)
    
    # Dataset
    splits = ['train', 'valid', 'test']
    allow_pattern_dict=[5,9,10,11]

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
    print('over')
    device_name = 1
    new_src,new_tgt = sample_add(dataset_dict,graph_samplers['train'],allow_pattern_dict,device_name)
    torch.save(new_src,f'./sampled_data/{args.dataname}/new_src_{device_name}.pt')
    torch.save(new_tgt,f'./sampled_data/{args.dataname}/new_tgt_{device_name}.pt')
def add_id():
    import torch
    import pandas as pd
    import re

    def number_to_pattern(input_str):
        elements = input_str.split()

        result = []
        for elem in elements:
            if elem.lstrip('-').isdigit():  # 检查是否是数字（包括负数）
                num = int(elem)
                if num < 0:
                    result.append('p')  # 负数变为 p
                else:
                    result.append('e')  # 正数变为 e
            else:
                result.append(elem)  # 非数字保持不变
    
        output_str = ' '.join(result)
        return output_str

# 读取 CSV 文件
    df = pd.read_csv('./akgr/metadata/pattern_filtered.csv')  # 替换为你的文件路径

# 定义处理函数
    def extract_letters(s):
        letters = re.findall(r'([a-z])', s)  # 提取所有小写字母
        return ' '.join(letters)  # 用空格连接

# 应用到 pattern_str 列
    df['cleaned_pattern'] = df['pattern_str'].apply(extract_letters)

# 查看结果
# print(df[['cleaned_pattern','id']])
    pattern_dict = df.set_index('cleaned_pattern')['id'].to_dict()
    print(pattern_dict)
# new_src = torch.load('new_src.pt')
    new_tgt = torch.load('new_tgt.pt')
    new_id = []
    for tgt in new_tgt:
        pattern = number_to_pattern(tgt)
        id = pattern_dict[pattern]
        new_id.append(id)
    torch.save(new_id,'new_id.pt')
# print(len(new_src))
# print(new_tgt[0:10])
if __name__ == '__main__':
    main()