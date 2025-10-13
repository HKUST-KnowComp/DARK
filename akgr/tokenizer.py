import sys, json
from tkinter import BASELINE

# sys.path.append('./utils/')
from akgr.utils.load_util import load_yaml
# from utils.load_util import load_yaml


from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, T5TokenizerFast, GPT2TokenizerFast, Qwen2TokenizerFast, BartTokenizerFast
import random
from transformers import AutoModel, AutoTokenizer


def get_vocab(special_tokens, offset, nentity, nrelation):
    vocab = {}
    vocab.update(special_tokens)
    for i in range(1, nentity+1): # [offset, offset + nentity - 1]
        vocab[str(i)] = offset + i - 1
    for i in range(1, nrelation+1): # [offset + nentity, offset + nentity + nrelation - 1]
        vocab[str(-i)] = offset + nentity + i - 1
    # vocab["-1"] = offset
    return vocab, offset + nentity + nrelation

def create_tokenizer(
        special_tokens: dict, offset: int,
        nentity: int, nrelation: int,
        is_gpt: bool):
    pre_tokenizer = WhitespaceSplit()
    vocab, vocab_size = get_vocab(special_tokens, offset=offset, nentity=nentity, nrelation=nrelation)
    model = WordLevel(vocab, unk_token='UNK')
    if not is_gpt:
        post_processor = TemplateProcessing(
            single='$0 END',
            # pair='$A START $B END',
            special_tokens=[('END', special_tokens['END'])]
        )
    else:
        post_processor = TemplateProcessing(
            single='$0 SEP',
            pair='$A SEP $B END',
            special_tokens=[('SEP', special_tokens['SEP']), ('END', special_tokens['END'])]
        )
    tokenizer = Tokenizer(model=model)

    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = post_processor
    # Just to let the tokenizer know about special tokens
    tokenizer.add_special_tokens(['START', 'END', 'PAD', 'UNK', 'SEP','EXPAND','MASK'])
    import io
    from contextlib import redirect_stdout
    trap = io.StringIO()
    with redirect_stdout(trap):
        if is_gpt:
            TokenizerFast = GPT2TokenizerFast
            tokenizer = TokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='START',
            eos_token='END',
            pad_token='END',
            unk_token='UNK',
            sep_token='SEP',
            ) # default padding side
        # tokenizer.pad_token = tokenizer.eos_token
        # 手动添加expand_token和mask_token
            tokenizer.expand_token = 'EXPAND'
            tokenizer.expand_token_id = tokenizer.convert_tokens_to_ids('EXPAND')
            tokenizer.mask_token = 'MASK'
            tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids('MASK')
        else:
            TokenizerFast = BartTokenizerFast
            tokenizer = TokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='START',
            eos_token='END',
            pad_token='END',
            unk_token='UNK',
            sep_token='SEP',
            mask_token='MASK',
            ) # default padding side
            tokenizer.expand_token = 'EXPAND'
            tokenizer.expand_token_id = tokenizer.convert_tokens_to_ids('EXPAND')
        
    return tokenizer, vocab_size

def create_custom_tokenizer_from_vocab(special_tokens, offset, nentity, nrelation):
    """
    Create a new tokenizer with custom vocabulary mapping
    
    Args:
        new_vocab (dict): Dictionary mapping tokens to IDs
        model_path (str): Path to the base model
    
    Returns:
        tokenizer: Modified tokenizer with custom vocabulary
    """
    pre_tokenizer = WhitespaceSplit()
    vocab, vocab_size = get_vocab(special_tokens, offset=offset, nentity=nentity, nrelation=nrelation)
    model = WordLevel(vocab, unk_token='UNK')
    tokenizer = Tokenizer(model=model)
    tokenizer.pre_tokenizer = pre_tokenizer
    # Just to let the tokenizer know about special tokens
    tokenizer.add_special_tokens(['START', 'END', 'UNK', 'SEP', 'EXPAND', 'MASK'])
    import io
    from contextlib import redirect_stdout
    trap = io.StringIO()
    with redirect_stdout(trap):
        TokenizerFast = Qwen2TokenizerFast
        tokenizer = TokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='START',
            eos_token='END',
            pad_token='END',
            unk_token='UNK',
            sep_token='SEP',
            )
        # 手动添加expand_token和mask_token
        tokenizer.expand_token = 'EXPAND'
        tokenizer.expand_token_id = tokenizer.convert_tokens_to_ids('EXPAND')
        tokenizer.mask_token = 'MASK'
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids('MASK')
    return tokenizer, vocab_size

def search_one_hop(source, graph,src_len):
    G = graph
    new_source_list = []
    for src in source:
            tmp = 0 
            count_list = []
            for node in src.split():
                node = int(node)
                tmp = tmp + 1
                in_edges = G.in_edges(node)  # 获取所有指向 node 的入边
                for (u,v,k) in in_edges:
                    count_list.append(u)
                    
            str_list = set(count_list)
            str_list = list(str_list)
            str_list = map(str, str_list)
            
            new_source = src +' '+ ' '.join(str_list)
            # print(new_source)
            split_list = new_source.split()
            truncated_list = split_list[:src_len]
            truncated_list = ' '.join(truncated_list)
            new_source_list.append(truncated_list)
    return new_source_list
            
           
def test_unify_extract_sample_to_device(tokenizer,sample, mask_number, model_type='sft_abd'):
        """Extract and process sample data for the model"""
        observation = sample['source']
        query = sample['target']
        pattern_id = sample['pattern_id']
         # For single sample, batch_size = 1
        sep_id = torch.full((1, 1), tokenizer.sep_token_id)
        bos_id = torch.full((1, 1), tokenizer.bos_token_id)
        eos_id = torch.full((1, 1), tokenizer.eos_token_id)
        # Create the custom format:  [bos] + query + [SEP] + observation + [eos] + padding
        mask_ids = torch.full((1, mask_number), tokenizer.mask_token_id)
        # Process single observation and query (not batch)
        if model_type == 'sft_ded':
            query_tokenized = tokenizer(
                query,
                add_special_tokens=False,
                return_tensors="pt")
            query_ids = query_tokenized["input_ids"]  # Shape: [1, seq_len]
            input_ids = torch.cat([bos_id, query_ids, sep_id, mask_ids, eos_id], dim=1)


        elif model_type == 'sft_abd':
            observation_tokenized = tokenizer(
                observation,
                add_special_tokens=False,
                return_tensors="pt")
            observation_ids = observation_tokenized["input_ids"]  # Shape: [1, seq_len]
            input_ids = torch.cat([bos_id, mask_ids,sep_id, observation_ids, eos_id], dim=1)

       
        return observation, query, pattern_id, input_ids

def test_unify_extract_sample_to_device_train(tokenizer, sample, model_type='unify', max_len=1024, max_delete=10, query_len=1024, answer_len=1024):
        """Extract and process sample data for the model"""
        observation = sample['source']
        query = sample['target']
        pattern_id = sample['pattern_id']
        
        # Create the custom format:  [bos] + query + [SEP] + observation + [eos] + padding
        
        # Process single observation and query (not batch)
        
        query_tokenized = tokenizer(
            query,
            add_special_tokens=False,
            return_tensors="pt")
        
        observation_tokenized = tokenizer(
            observation,
            add_special_tokens=False,
            return_tensors="pt")
        
        observation_ids = observation_tokenized["input_ids"]  # Shape: [1, seq_len]
        query_ids = query_tokenized["input_ids"]  # Shape: [1, seq_len]

        observation_length = observation_ids.shape[1]
        query_length = query_ids.shape[1]

        # For single sample, batch_size = 1
        sep_id = torch.full((1, 1), tokenizer.sep_token_id)
        bos_id = torch.full((1, 1), tokenizer.bos_token_id)
        eos_id = torch.full((1, 1), tokenizer.eos_token_id)
        
        
        
        if model_type == 'unify':

            if max_len -query_length-observation_length-2 > 0 and max_delete > 0:
                eos_count = torch.randint(
                    low=0,
                    high=min(max_delete, max_len -query_length-observation_length-2),
                    size=(1,),
                ).item()
                eos_tensor = torch.full((1, eos_count), tokenizer.eos_token_id, dtype=query_ids.dtype)
            else:
                eos_count = 0
                eos_tensor = torch.empty((1, 0), dtype=query_ids.dtype)

            # Concatenate query + sep + observation
            middle_ids = torch.cat([query_ids, sep_id, observation_ids, eos_tensor], dim=1)
            input_ids = torch.cat([bos_id, middle_ids, eos_id], dim=1)
            attention_mask = torch.cat([torch.zeros((1, 1)), torch.ones_like(middle_ids), torch.zeros((1, 1))], dim=1)
        elif model_type == 'sft_abd':
            if query_len - query_length - 1> 0 and max_delete > 0:
                eos_count = torch.randint(
                    low=0,
                    high=min(max_delete, query_len - query_length - 1),
                    size=(1,),
                ).item()
                eos_tensor_query = torch.full((1, eos_count), tokenizer.eos_token_id, dtype=query_ids.dtype)
            else:
                eos_count = 0
                eos_tensor_query = torch.empty((1, 0), dtype=query_ids.dtype)
            middle_ids = torch.cat([query_ids, eos_tensor_query], dim=1)
            end_ids = torch.cat([sep_id, observation_ids, eos_id], dim=1)
            input_ids = torch.cat([bos_id, middle_ids, end_ids], dim=1)
            attention_mask = torch.cat([torch.zeros_like(bos_id), torch.ones_like(middle_ids), torch.zeros_like(end_ids)], dim=1)
        
        elif model_type == 'sft_ded':
            if answer_len - observation_length - 1 > 0 and max_delete > 0:
                eos_count = torch.randint(
                low=0,
                high=min(max_delete, answer_len - observation_length - 1),
                size=(1,),
                ).item()
                eos_tensor_observation = torch.full((1, eos_count), tokenizer.eos_token_id, dtype=query_ids.dtype)
            else:
                eos_count = 0
                eos_tensor_observation = torch.empty((1, 0), dtype=query_ids.dtype)

            middle_ids = torch.cat([ observation_ids, eos_tensor_observation], dim=1)
            start_ids = torch.cat([bos_id, query_ids, sep_id], dim=1)
            input_ids = torch.cat([start_ids, middle_ids, eos_id], dim=1)
            attention_mask = torch.cat([torch.zeros_like(start_ids), torch.ones_like(middle_ids), torch.zeros((1, 1))], dim=1)
        

        # Calculate current length and padding needed
        current_length = input_ids.shape[1]
        max_length = max_len
        
        if current_length < max_length:
            # Calculate padding length
            padding_length = max_length - current_length
            
            # Create padding tokens using eos_token_id
            padding_ids = torch.full((1, padding_length), tokenizer.eos_token_id)
            padding_mask = torch.zeros((1, padding_length))
            
            # Concatenate padding to the right
            input_ids = torch.cat([input_ids, padding_ids], dim=1)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=1)
        
        return observation, query, pattern_id, input_ids, attention_mask
def test_unify_extract_sample_to_device_length(tokenizer, sample, model_type='unify', query_len=1024, answer_len=1024):
        """Extract and process sample data for the model"""
        observation = sample['source']
        query = sample['target']
        pattern_id = sample['pattern_id']
        batch = len(observation)
        # Create the custom format:  [bos] + query + [SEP] + observation + [eos] + padding
        
        # Process single observation and query (not batch)
        
        query_tokenized = tokenizer(
            query,
            add_special_tokens=False,
            max_length=query_len,
            padding='max_length',
            return_tensors="pt")
        
        observation_tokenized = tokenizer(
            observation,
            add_special_tokens=False,
            max_length=answer_len,
            padding='max_length',
            return_tensors="pt")
        
        observation_ids = observation_tokenized["input_ids"]  # Shape: [1, seq_len]
        query_ids = query_tokenized["input_ids"]  # Shape: [1, seq_len]

        observation_length = observation_ids.shape[1]
        query_length = query_ids.shape[1]

        # For single sample, batch_size = 1
        sep_id = torch.full((batch, 1), tokenizer.sep_token_id)
        bos_id = torch.full((batch, 1), tokenizer.bos_token_id)
        eos_id = torch.full((batch, 1), tokenizer.eos_token_id)
        
        if model_type == 'unify':
            input_ids = torch.cat([bos_id, query_ids, sep_id, observation_ids, eos_id], dim=1)
        
            attention_mask = torch.cat([torch.zeros((batch, 1)), torch.ones_like(query_ids), torch.zeros((batch, 1)), torch.ones_like(observation_ids), torch.zeros((batch, 1))], dim=1)
            
        elif model_type == 'sft_abd':
            mask_ids = torch.full((batch, query_length), tokenizer.mask_token_id)
            input_ids = torch.cat([bos_id, mask_ids, sep_id, observation_ids, eos_id], dim=1)
            attention_mask = torch.cat([torch.zeros_like(bos_id), torch.ones_like(query_ids), torch.zeros((batch, 1)), torch.zeros_like(observation_ids), torch.zeros((batch, 1))], dim=1)
            
        elif model_type == 'sft_ded':
            mask_ids = torch.full((batch, observation_length), tokenizer.mask_token_id)
            input_ids = torch.cat([bos_id, query_ids, sep_id, mask_ids, eos_id], dim=1) 
            attention_mask = torch.cat([torch.zeros_like(bos_id), torch.zeros_like(query_ids), torch.zeros((batch, 1)), torch.ones_like(observation_ids), torch.zeros((batch, 1))], dim=1)

        
        return observation, query, pattern_id, input_ids, attention_mask
import torch
def unify_extract_sample_to_device(device,
        sample, tokenizer, is_gpt:bool,
        src_len, tgt_len, is_gen:bool):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    if not is_gpt:
        print('not support')
        
    else:
        # Create the custom format: padding + [START] + src + [SEP] + tgt + [START] + padding
        # Save original padding side
        original_padding_side = tokenizer.padding_side
        
        # Step 1: Create padding + [START] + src (right padding)
        tokenizer.padding_side = 'left'
        source_with_start = ['START ' + s for s in source]
        source_tokenized = tokenizer(
            source_with_start,
            padding='max_length',
            max_length=src_len + 1,  # +1 for START token
            add_special_tokens=False,
            return_tensors="pt").to(device)
        
        # Step 2: Create tgt + [START] + padding (left padding)
        tokenizer.padding_side = 'right'
        target_with_start = [t + 'START' for t in target]
        target_tokenized = tokenizer(
            target_with_start,
            padding='max_length',
            max_length=tgt_len + 1,  # +1 for START token
            add_special_tokens=False,
            return_tensors="pt").to(device)
        source_ids = source_tokenized["input_ids"]
        source_mask = source_tokenized["attention_mask"]

        target_ids = target_tokenized["input_ids"]
        target_mask = target_tokenized["attention_mask"]
        
        batch_size = len(source)
        sep_id = torch.full((batch_size, 1), tokenizer.sep_token_id, device=device)
        sep_mask = torch.ones((batch_size, 1), device=device)
        
        input_ids = torch.cat([source_ids, sep_id, target_ids], dim=1)
        attention_mask = torch.cat([source_mask, sep_mask, target_mask], dim=1)

        # 重新计算合并后的source_mask和target_mask
        source_len = source_ids.shape[1]
        target_len = target_ids.shape[1]
        
        if is_gen:
            # source_attention_mask: source + sep 部分
            source_attention_mask = torch.zeros_like(attention_mask)
            source_attention_mask[:, :source_len+1] = 1  # source + sep 部分

            # merged_target_mask: sep + target 部分（包括 padding）
            target_attention_mask = torch.zeros_like(attention_mask)
            target_attention_mask[:, source_len:] = 1  # sep + target 部分
        else:
            # source_attention_mask: source + sep 部分
            source_attention_mask = torch.zeros_like(attention_mask)
            source_attention_mask[:, :source_len+1] = 1  # source + sep 部分

            # merged_target_mask: sep + target 部分（包括 padding）
            target_attention_mask = torch.zeros_like(attention_mask)
            target_attention_mask[:, source_len:] = 1  # sep + target 部分

        labels = None
    

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_attention_mask

def new_extract_sample_to_device(device,
        sample, tokenizer, is_gpt:bool,
        src_len, tgt_len, is_gen:bool):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    if not is_gpt:
        print('not support')
        
    else:
        source_target_tokenized = tokenizer(
            source, target,
            padding='longest',
            # max_length=src_len+tgt_len,
            return_tensors="pt").to(device)
        # labels is the source SEP target END, ...
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # ... with the source part's loss ignored
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if is_gen == False: # (train/valid) input = source SEP target END, default padding side
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
    
        else: # (test/optimize) input = source c, left padding (align the last tokens to the right)
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                source,
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask

        # labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask



def debug():
    config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    offset = config_dataloader['offset']
    special_tokens = config_dataloader['special_tokens']
    tokenizer, _ = create_tokenizer(special_tokens, offset, nentity=200000, nrelation=2000, is_gpt=True)
    sample1 = {'answers': [1, 2, 3, 4], "query": ["(","i","(","n","(","p","(",-1,")","(","p","(",0,")","(","e","(",0,")",")",")",")",")","(","p","(",-567,")","(","e","(",24623,")",")",")",")"], "pattern_str":"(i,(n,(p,(p,(e)))),(p,(e)))"}
    sample2 = {'answers': [1, 2], "query": ["(", "p", "(", -1, ")", "(", "e", "(", 0, ")", ")", ")"], "pattern_str": "(p,(e))"}
    from utils.parsing_util import qry_shift_indices, ans_shift_indices, qry_str_2_actionstr
    def list_to_str(l: list) -> str:
        # print('before', l)
        # print('after', ' '.join([str(x) if isinstance(x, int) else x for x in l]))
        return ' '.join([str(x) if isinstance(x, int) else x for x in l])
    sample = {}
    sample['source'] = [list_to_str(ans_shift_indices(sample1['answers'])), list_to_str(ans_shift_indices(sample2['answers']))]
    sample['target'] = [qry_str_2_actionstr(list_to_str(qry_shift_indices(sample1['query']))), qry_str_2_actionstr(list_to_str(qry_shift_indices(sample2['query'])))]
    sample['pattern_id'] = [1, 2]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
        new_extract_sample_to_device(device, sample, tokenizer, is_gpt=True, src_len=33, tgt_len=66, is_gen=True)
    # input_ids = tokenizer(sample['source'], padding='max_length', max_length=33, return_tensors="pt")
    print('----')
    print('source')
    print(source)
    print('target')
    print(target)
    print('input_ids')
    print(input_ids)
    print('attention_mask')
    print(attention_mask)
    print('labels')
    print(labels)
    labels[labels == -100] = 0
    print(tokenizer.batch_decode(labels, skip_special_tokens=True))


    source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
        new_extract_sample_to_device(device, sample, tokenizer, is_gpt=True, src_len=33, tgt_len=33, is_gen=False)
    # input_ids = tokenizer(sample['source'], padding='max_length', max_length=33, return_tensors="pt")
    print('source')
    print(source)
    print('target')
    print(target)
    print('input_ids')
    print(input_ids)
    print('attention_mask')
    print(attention_mask)
    print(source_attention_mask)
    print('labels')
    print(labels)
    labels[labels == -100] = 0
    print(tokenizer.batch_decode(labels, skip_special_tokens=True))

# def source_to_prompt(sample,args):
#     source = sample['source']              # 单个字符串，如 "19346"
#     target = sample['target']              # 单个目标
#     condition = getattr(args, 'condition', 'unconditional')

#     if condition == 'unconditional':
#         sample['prompt'] = source
#     elif condition == 'pattern':
#         target_pattern = number_to_pattern(target)
#         sample['prompt'] = f"{source} [SEP] {target_pattern}"
#     elif condition == 'relationnumber':
#         sample['prompt'] = f"{source} [SEP] {number_to_epnumber(target)[1]}"
#     elif condition == 'entitynumber':
#         sample['prompt'] = f"{source} [SEP] {number_to_epnumber(target)[0]}"
#     elif condition == 'relation':
#         sample['prompt'] = f"{source} [SEP] {number_to_epspecific(target)[1]}"
#     elif condition == 'entity':
#         sample['prompt'] = f"{source} [SEP] {number_to_epspecific(target)[0]}"
#     else:
#         raise ValueError(f"Unsupported condition: {condition}")
#     return sample
if __name__ == '__main__':
    debug()