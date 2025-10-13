from math import e
import os, sys, argparse, warnings
import random
import numpy as np
import pandas as pd



def set_seed(seed=42):
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variables for full determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")
    
   
import swanlab
from swanlab.integration.transformers import SwanLabCallback

    
import json
import yaml
import pandas as pd
import torch
from tqdm import tqdm
import random
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
# dataloader
from akgr.dataloader import new_create_dataset
from akgr.tokenizer import create_custom_tokenizer_from_vocab, create_tokenizer, test_unify_extract_sample_to_device, test_unify_extract_sample_to_device_train, test_unify_extract_sample_to_device_length

from akgr.abduction_model.model.custom_gpt import create_gpt2_transformer, DiffusionModel

# utils
from akgr.utils.stat_util import stat_scores_by_pattern
from akgr.utils.load_util import load_yaml, load_model, save_model, load_and_filter_query_patterns
from akgr.kgdata import load_kg

# evaluation
from akgr.evaluation import scoring_input_wordlist_batch, scoring_input_act_batch
from akgr.evaluation_deductive import evaluate_deductive, evaluate_deductive_batch
from akgr.utils.parsing_util import qry_actionprefix_get_branching, is_strint
from akgr.abduction_model.model.custom_mydream.generation_expand_utils import DreamGenerationConfig
# Custom trainer
from akgr.abduction_model.trainer import AbductionTrainer, AbductionTrainingArguments
from akgr.kgdata.custom_dataset import AbductionDataset, AbductionDataset_length, GRPODataset   
from akgr.abduction_model.model.custom_mydream.modeling_dream import DreamModel, DreamConfig
from akgr.abduction_model.model.custom_mydream.modeling_dream_length import DreamModel_length
# rl
from akgr.abduction_model.custom_grpoconfig import GRPOConfig

from accelerate import Accelerator
import logging
from verl.utils.model import compute_position_id_with_mask
import torch.nn as nn

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Safer multiprocessing start method for multi-GPU
try:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
except Exception:
    pass

# Global variables
raw_dataset = None
device = None
pattern_filtered = None
nentity = None
nrelation = None
offset = None
special_tokens = None
do_correction = False
graph_samplers = None




def reward_func(query_list, answer_list, step, **kwargs):
   
    # you have a query, and an answer, you need to cacluate the jaccard score, no meaning for pred_word_batch
    scores, failures_batch_id = scoring_input_act_batch(
            pred_word_batch=query_list,
            label_word_batch=query_list,
            ans_word_batch=answer_list,
            scoring_method=['jaccard'],
            do_correction=do_correction,
            graph_samplers=graph_samplers,
            searching_split='train',
            return_failures=True,
            )
    rewards=[torch.tensor(score['jaccard'], dtype=torch.float) for score in scores]
    # 记录step和对应的reward
    reward_values = [r.item() for r in rewards]
    logger.info(f"Step {step}:  Avg Reward = {np.mean(reward_values):.4f},std Reward = {np.std(reward_values):.4f}")
    return rewards

def create_training_args(args, config_train, training_mode):
    """Create training arguments for the Trainer"""
    output_dir = os.path.join(args.checkpoint_root, args.modelname, 
                             f'{args.dataname}-{training_mode}-{args.merge_prob}')
    
    training_args = AbductionTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config_train['nepoch'],
        per_device_train_batch_size=config_train.get('batch_size', 8),
        per_device_eval_batch_size=config_train.get('batch_size', 8),
        learning_rate=float(config_train["lr"]),
        warmup_steps=config_train["warm_up"],
        weight_decay=float(config_train["weight_decay"]),
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        save_total_limit=args.save_frequency,
        eval_strategy="epoch" if args.do_valid else "no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        # 更激进的多卡安全配置
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        group_by_length=False,
        ddp_find_unused_parameters=False,
        # 禁用所有可能导致卡死的功能
        dataloader_drop_last=False,
        dataloader_prefetch_factor=None,
        # 设置更保守的DDP参数
        ddp_bucket_cap_mb=25,
        ddp_broadcast_buffers=False,
        # 禁用 wandb
        report_to=['wandb'],
        # Custom parameters
        training_mode=training_mode,
    )
    
    return training_args

def train_with_trainer(args, model, tokenizer, dataset_dict, config_train,  qry_len, ans_len):
    """Train the model using HuggingFace Trainer"""
    
    if args.training_mode == 'unify':
        # Single training mode
        training_args = create_training_args(args, config_train, 'unify')
        if args.merge_prob != 0.0:
        # Create datasets
            train_dataset = AbductionDataset(
                dataset_dict['train'], tokenizer, qry_len, ans_len, 'unify'
                )
        else:
            print('use AbductionDataset_length')
            train_dataset = AbductionDataset_length(
            dataset_dict['train'], tokenizer, qry_len, ans_len, 'unify'
        )
        eval_dataset = None
        if args.do_valid:
            eval_dataset = AbductionDataset(
                dataset_dict['valid'], tokenizer, qry_len, ans_len, 'unify'
            )
    else:
        training_args = create_training_args(args, config_train, 'sft')
        
        # Create datasets
        if args.merge_prob != 0.0:
            train_dataset = AbductionDataset(
                dataset_dict['train'], tokenizer, qry_len, ans_len, 'sft'
            )
        else:
            print('use AbductionDataset_length')
            train_dataset = AbductionDataset_length(
            dataset_dict['train'], tokenizer, qry_len, ans_len, 'sft'
        )
        eval_dataset = None
        if args.do_valid:
            eval_dataset = AbductionDataset(
                dataset_dict['valid'], tokenizer, qry_len, ans_len, 'sft'
            )
        # 多卡训练配置
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
            print(f"Enable multi-GPU training, GPUs: {num_gpus}")
            per_device_batch_size = config_train.get('batch_size', 8)
            training_args.per_device_train_batch_size = per_device_batch_size
            training_args.per_device_eval_batch_size = per_device_batch_size
            print(f"Per-GPU batch size: {per_device_batch_size}")
            
            # 
            training_args.dataloader_pin_memory = False
            training_args.remove_unused_columns = False
            training_args.ddp_find_unused_parameters = False
            training_args.ddp_bucket_cap_mb = 25
        
        # Create trainer
    trainer = AbductionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            custom_processing_class=tokenizer,
            name=args.modelname,
            attention_all = args.attention_all,
            merge_prob = args.merge_prob
        
        )
       
        # Train
    trainer.train()
        
        # Save final model
    final_checkpoint = os.path.join(training_args.output_dir, 'final')
    trainer.save_model(final_checkpoint)
        
    
def test(args, model, tokenizer, dataset_dict, graph_samplers, mask_number, test_type):  
    niter = len(dataset_dict)  
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    expand_token_id = tokenizer.expand_token_id
    generation_config = DreamGenerationConfig(mask_token_id=mask_token_id, 
                                                pad_token_id=eos_token_id, 
                                                bos_token_id=bos_token_id, 
                                                eos_token_id=eos_token_id,
                                                expand_token_id=expand_token_id,
                                                delete_token_id=eos_token_id,
                                                max_length = 50
                                                )
    
    scores_all = []
    pattern_id_all = []
    
  
    os.makedirs(f"results/{args.dataname}", exist_ok=True)
    
    csv_file_path = f"results/{args.dataname}/{test_type}_detailed.csv"
    
    for iter, sample in (pbar := tqdm(enumerate(dataset_dict), total=niter)):
        # if iter == 1000:
        #     break
        if args.merge_prob == 0.5:
            observation, query, pattern_id, input_ids = test_unify_extract_sample_to_device(tokenizer, sample, mask_number, model_type=test_type)
        else:
            observation, query, pattern_id, input_ids, need_to_mask = test_unify_extract_sample_to_device_length(tokenizer, sample, query_len=10, answer_len=32, model_type=test_type)
      
        input_ids = input_ids.to(device)

        model.to(device)
        output = model.diffusion_generate(
            inputs=input_ids,
            generation_config=generation_config,
            temperature=0.1,
            alg = 'entropy',
            alg_temp = 0,
            top_p = 0.9,
            return_dict_in_generate = True,
            output_history = True,
            number_transfer_tokens = 1
        )
        
        test_ids = output.sequences
        history = output.history
        # print(f"test_ids: {test_ids}")
   
        sep_token_id = tokenizer.sep_token_id

      
        sep_positions = (test_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]  


        if len(sep_positions) > 0:
            sep_pos = sep_positions[0].item()
            left_part = test_ids[0, :sep_pos]  
            right_part = test_ids[0, sep_pos+1:]  
            query_decode = tokenizer.decode(left_part, skip_special_tokens=True)
            observation_decode = tokenizer.decode(right_part, skip_special_tokens=True)
    
            # print(f"query_decode: {query_decode}")
            # print(f"observation_decode: {observation_decode}")
        
        
        if test_type == 'sft_abd':
            scoring_fn = scoring_input_act_batch 
            scores, failures_batch_id = scoring_fn(
                    pred_word_batch=[query_decode],
                    label_word_batch=[query],
                    ans_word_batch=[observation],
                    scoring_method=['smatch', 'jaccard'] + ['count0'] * (args.test_count0 == True),
                    do_correction=args.do_correction,
                    graph_samplers=graph_samplers,
                    searching_split=args.test_split,
                    return_failures=True,
                    verbose=args.vs)
            
            
            
            gathered_scores = scores
            gathered_pattern_id = [pattern_id]

            
            scores_all.extend(gathered_scores)
            pattern_id_all.extend(gathered_pattern_id)
            score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
                # print(score_df)
            pbar.set_description(f's: {round(score_df.loc["all",("smatch","mean")], 4)}, j: {round(score_df.loc["all",("jaccard","mean")], 4)}')
            scores_path = csv_file_path
            score_df.to_csv(scores_path)

        elif test_type == 'sft_ded':
            scores = evaluate_deductive(source_list=[observation], pred_list=[observation_decode], nentity=nentity)        
            
           
            score_dict = {
                'jaccard': scores['avg_jaccard'],
                'mrr': scores['avg_mrr'],
                'hit_at_1': scores['avg_hit_at_1'],
                'hit_at_3': scores['avg_hit_at_3'],
                'hit_at_10': scores['avg_hit_at_10']
            }
            
            scores_all.append(score_dict)
            pattern_id_all.append(pattern_id)
            score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
                # print(score_df)
            pbar.set_description(f'mrr: {round(score_df.loc["all",("mrr","mean")], 4)}, h1: {round(score_df.loc["all",("hit_at_1","mean")], 4)}, h3: {round(score_df.loc["all",("hit_at_3","mean")], 4)}, h10: {round(score_df.loc["all",("hit_at_10","mean")], 4)}')
            scores_path = csv_file_path
            score_df.to_csv(scores_path)
            
    
    return 0

def test_batch(args, model, tokenizer, dataset_dict, graph_samplers, test_type):  
    niter = len(dataset_dict)  
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    expand_token_id = tokenizer.expand_token_id
    sep_token_id = tokenizer.sep_token_id
    generation_config = DreamGenerationConfig(mask_token_id=mask_token_id, 
                                                pad_token_id=eos_token_id, 
                                                bos_token_id=bos_token_id, 
                                                eos_token_id=eos_token_id,
                                                expand_token_id=expand_token_id,
                                                delete_token_id=eos_token_id,
                                                sep_token_id=sep_token_id,
                                                max_length = 46,
                                                )
    
   
    scores_all = []
    pattern_id_all = []
    

    os.makedirs(f"results/{args.dataname}/{args.merge_prob}-{args.explore_ratio}-{args.deductive_ratio}", exist_ok=True)
    

    csv_file_path = f"results/{args.dataname}/{args.merge_prob}-{args.explore_ratio}-{args.deductive_ratio}/{test_type}_detailed.csv"

    batch_size = 64 
    for iter in (pbar := tqdm(range(0, niter, batch_size), total=(niter + batch_size - 1) // batch_size)):
        # if iter >= 1000:
        #     break
        
      
        batch_samples = dataset_dict[iter:iter + batch_size]
        
        
        observation, query, pattern_id, input_ids, need_to_mask = test_unify_extract_sample_to_device_length(tokenizer, batch_samples, query_len=10, answer_len=32, model_type=test_type)
            
       
        input_ids = input_ids.to(device)
        # print(f"query: {query}")
        # print(f"observation: {observation}")
        # print(f"input_ids: {input_ids}")
        
        model.to(device)    
        output = model.diffusion_generate(
            inputs=input_ids,
            generation_config=generation_config,
            temperature=0.1,
            alg = 'entropy',
            alg_temp = 0,
            top_p = 0.9,
            return_dict_in_generate = True,
            steps=64,
            output_history = True,
            number_transfer_tokens = 1
        )
        
        test_ids = output.sequences
        history = output.history
       
        sep_token_id = tokenizer.sep_token_id

        sep_positions = (test_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]  

        
        sep_pos = sep_positions[0].item()
           
        left_parts = test_ids[:, :sep_pos]  
        right_parts = test_ids[:, sep_pos+1:]  
            

        query_decodes = [tokenizer.decode(part, skip_special_tokens=True) for part in left_parts]
        observation_decodes = [tokenizer.decode(part, skip_special_tokens=True) for part in right_parts]
    
        # print(f"observation: {observation}")
        # print(f"observation_decodes: {observation_decodes}")
        
        
        if test_type == 'sft_abd':
            
            scoring_fn = scoring_input_act_batch 
            scores, failures_batch_id = scoring_fn(
                    pred_word_batch=query_decodes,
                    label_word_batch=query,
                    ans_word_batch=observation,
                    scoring_method=['smatch', 'jaccard'] + ['count0'] * (args.test_count0 == True),
                    do_correction=args.do_correction,
                    graph_samplers=graph_samplers,
                    searching_split=args.test_split,
                    return_failures=True,
                    verbose=args.vs)
            gathered_scores = scores
            gathered_pattern_id = pattern_id

            
            scores_all.extend(gathered_scores)
            pattern_id_all.extend(gathered_pattern_id)
            score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
                # print(score_df)
            pbar.set_description(f's: {round(score_df.loc["all",("smatch","mean")], 4)}, j: {round(score_df.loc["all",("jaccard","mean")], 4)}')
            scores_path = csv_file_path
            score_df.to_csv(scores_path)

        elif test_type == 'sft_ded':
            scores = evaluate_deductive_batch(source_list=observation, pred_list=observation_decodes, nentity=nentity)
            gathered_scores = scores
            gathered_pattern_id = pattern_id
            
            scores_all.extend(gathered_scores)
            pattern_id_all.extend(gathered_pattern_id)
            score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
                # print(score_df)
            pbar.set_description(f'mrr: {round(score_df.loc["all",("mrr","mean")], 4)}, h1: {round(score_df.loc["all",("hit_at_1","mean")], 4)}, h3: {round(score_df.loc["all",("hit_at_3","mean")], 4)}, h10: {round(score_df.loc["all",("hit_at_10","mean")], 4)}')
            scores_path = csv_file_path
            score_df.to_csv(scores_path)
    return 0


def optimize(args, configs, dataset_dict, model, tokenizer):
    
    print('GRPO Setting Up')
    # Prepare dataset
    
    dataset = GRPODataset(dataset_dict)
    
    output_dir = os.path.join(args.checkpoint_root, args.modelname, 
                             f'{args.dataname}-grpo-{args.merge_prob}-{args.explore_ratio}-{args.deductive_ratio}-continue')
    swanlab_callback = SwanLabCallback(
    project="trl", 
    experiment_name="GRPO"
        )  
    grpo_config = GRPOConfig(
        seed=args.seed, # default
        output_dir= output_dir,
        num_train_epochs=configs['epochs'],
        learning_rate=float(configs['lr']),
        beta=configs['rl_init_kl_coef'], #do not knows
        epsilon=configs['rl_cliprange'],
        save_total_limit=args.save_frequency,
        num_generations=configs['num_generations'],
        eval_strategy="no",
        save_strategy="epoch",
        per_device_train_batch_size=configs['batch_size'],
        per_device_eval_batch_size = configs['batch_size'],
        remove_unused_columns=False, # Important. By default, it removes unrecognized columns if hf dataset is passed
        report_to='none',
        random_masking=True,
        num_iterations=2,
        scale_rewards=True,
        explore_ratio=args.explore_ratio,
        )
    if args.merge_prob == 0.0:
        from akgr.abduction_model.custom_grpo2 import DiffuGRPOTrainer
    else:
        raise ValueError(f"Merge prob {args.merge_prob} is not supported")
    trainer = DiffuGRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[swanlab_callback]
        )
    trainer.train()
    final_checkpoint = os.path.join(grpo_config.output_dir, 'final')
    trainer.save_model(final_checkpoint)
    return 0

def load_model_by_mode(args, tokenizer, ntoken):
    """Load model based on mode"""
    if args.resume_epoch == 0 and args.training_mode == 'unify' and args.mode == 'training':
       
        if args.modelname == 'dreamon':
            model_path = "Dream-org/DreamOn-v0-7B"
            model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
            model.resize_token_embeddings(ntoken+1)
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
        elif args.modelname == 'mydream':
            config = DreamConfig(vocab_size=ntoken+1,mask_token_id=tokenizer.mask_token_id,pad_token_id=tokenizer.pad_token_id)
            model = DreamModel(config)
        elif args.modelname == 'gpt2':
            model_path = "gpt2"
            model = create_gpt2_transformer(ntoken, tokenizer, model_path)

    elif args.resume_epoch == 0 and args.training_mode == 'sft' and args.mode == 'training':
        model_path = os.path.join(args.checkpoint_root, args.modelname,
                     f'{args.dataname}-unify-{args.merge_prob}','final')  
        print(f'Loading {args.modelname} model from trainer checkpoint: {model_path}')
        if args.modelname == 'gpt2':
            model = DiffusionModel.from_pretrained(model_path)
        elif args.modelname == 'mydream':
            model = DreamModel.from_pretrained(model_path)
        else:
            model = AutoModel.from_pretrained(model_path)
        print(f'Successfully loaded {args.modelname} model from {model_path}')
       
    elif args.mode in ['testing']:
        model_path = os.path.join(args.checkpoint_root, args.modelname,
                             f'{args.dataname}-{args.training_mode}-{args.merge_prob}','final')
        print(f'Loading {args.modelname} model from trainer checkpoint: {model_path}')
        if args.modelname == 'gpt2':
            model = DiffusionModel.from_pretrained(model_path)
        elif args.modelname == 'mydream':
            if args.merge_prob == 0.0:
                model = DreamModel_length.from_pretrained(model_path)
            else:
                model = DreamModel.from_pretrained(model_path)
        else:
            model = AutoModel.from_pretrained(model_path)
        print(f'Successfully loaded {args.modelname} model from {model_path}')

    if args.mode in ['optimizing']:
    
        model_path = os.path.join(args.checkpoint_root, args.modelname, 
                             f'{args.dataname}-sft-{args.merge_prob}','final')
        if args.modelname == 'gpt2':
            model = DiffusionModel.from_pretrained(model_path)
        elif args.modelname == 'mydream':
            if args.merge_prob == 0.0:
                model = DreamModel_length.from_pretrained(model_path)
            else:
                model = DreamModel.from_pretrained(model_path)

        else:
            model = AutoModel.from_pretrained(model_path)
        print(f'Successfully loaded {args.modelname} model from {model_path}')

    #     lora_config = LoraConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         r=8,  
    #         lora_alpha=16,  
    #         lora_dropout=0.0,  
    #         target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  
    #         bias="none",
    # )


    #     model = get_peft_model(model, lora_config)

    #       
    #     model.print_trainable_parameters()
    

    return model

def my_parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument('--modelname', default='mydream')
    parser.add_argument('--config-dataloader', default='akgr/configs/config-dataloader.yml')
    parser.add_argument('--config-train', default='akgr/configs/config-train.yml')
    parser.add_argument('--config-model', default='akgr/configs/config-model.yml')
    parser.add_argument('--config-batchsize', default='akgr/configs/config-batchsize.yml')
    parser.add_argument('--config-rl', default='akgr/configs/config-grpo.yml')
    parser.add_argument('--overwrite_batchsize', type=int, default=0)

    # Data
    parser.add_argument('--data_root', default='/home/zhangziwei/gys/AbductiveKGR/sampled_data')
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--scale', default='full')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)

    # Checkpoint
    parser.add_argument('--checkpoint_root', default='./checkpoints/')
    parser.add_argument('-r', '--resume_epoch', type=int, default=0)

    parser.add_argument('--vs', action='store_true', help='verbose flag for smatch result')
    parser.add_argument('--do_correction', action='store_true', help='verbose flag for smatch result')

    parser.add_argument('--drop', type=float, default=0.)

    # Training
    parser.add_argument('--training_mode', default='unify')
    parser.add_argument('--attention_all', action='store_true', help='verbose flag for smatch result')
    parser.add_argument('--merge_prob', type=float, default=0.5)
    # Validation
    parser.add_argument('--do_valid', action='store_true', help='verbose flag for smatch result')
    parser.add_argument('--do_initial_valid', action='store_true', help='verbose flag for smatch result')

    #optimizing
    parser.add_argument('--explore_ratio', type=float, default=0.5)
    parser.add_argument('--deductive_ratio', type=float, default=0.5)
    # Testing
    parser.add_argument('--test_proportion', type=float, default=1)
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--test_top_k', type=int, default=0)
    parser.add_argument('--test_count0', action='store_true')

    parser.add_argument('--save_frequency', type=int, default=1)

    parser.add_argument('--mode', default='training')
    parser.add_argument('--accelerate', action='store_true')
    parser.add_argument('--constrained', action='store_true')
    parser.add_argument('--compare_training_step', action='store_true', help='Compare training_step results with sampling results')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    args = parser.parse_args()
    return args

def main():
    args = my_parse_args()
    print(f'# Running main.py in {args.mode} mode with:')
    print(f'args:\n{args}\n')

    # Set random seed for reproducibility
    set_seed(args.seed)
    if args.mode == 'optimizing':
    
        log_dir = f'logs/{args.dataname}/{args.merge_prob}-{args.explore_ratio}-{args.deductive_ratio}'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'logging.log')
    
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            print(f"Logging to file: {log_file}")
    
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
    global graph_samplers
    graph_samplers = kg.graph_samplers

    # Device
    global device
    if args.accelerate and args.mode != 'optimizing':

        if 'MASTER_PORT' not in os.environ:
            import random
            port = random.randint(29500, 30000)
            os.environ['MASTER_PORT'] = str(port)
        
        accelerator = Accelerator()
        device = accelerator.device
        
       
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        accelerator = None
    print(f'DEVICE: {device}')

    # Model information
    model_name = args.modelname
    if args.modelname == 'dreamon':
        args.model_path = 'Dream-org/DreamOn-v0-7B'
    elif args.modelname == 'dream':
        args.model_path = 'Dream-org/Dream-v0-Instruct-7B'
    elif args.modelname == 'gpt2':
        args.model_path = 'gpt2'
    elif args.modelname == 'mydream':
        args.model_path = 'mydream'
    else:
        raise ValueError(f'Model {args.modelname} not supported')
   
    qry_len = config_dataloader['qry_len'] 
    ans_len = config_dataloader['ans_len'] 
    print(f'model_name:{model_name}\n')

    

    print('=' * 50)

    # Dataset
    if args.mode == 'training':  
        splits = ['train', 'valid']
    elif args.mode == 'testing': 
        splits = [args.test_split]
    elif args.mode == 'optimizing': 
        splits = ['train']

    print('Creating dataset & dataloader')
    global nentity, nrelation
    
    dataset_dict, nentity, nrelation = new_create_dataset(
            dataname=args.dataname,
            scale=args.scale,
            answer_size=args.max_answer_size,
            pattern_filtered=pattern_filtered,
            data_root=args.data_root,
            splits=splits,
        )
    print(nentity, nrelation)
    if args.mode == 'testing' and args.test_proportion < 1:
        nrows = dataset_dict[args.test_split].shape[0]
        dataset_dict[args.test_split] = dataset_dict[args.test_split].select(
            random.sample(range(nrows), int(nrows * args.test_proportion))
        )
    # Tokenizer
    print('Creating tokenizer')
    if args.modelname == 'gpt2' or args.modelname == 'mydream':
        tokenizer,ntoken = create_tokenizer(
            special_tokens=special_tokens,
            offset=offset,
            nentity=nentity,
            nrelation=nrelation,
            is_gpt=True
        )
    else:
        tokenizer, ntoken = create_custom_tokenizer_from_vocab(
        special_tokens=special_tokens,
        offset=offset,
        nentity=nentity,
        nrelation=nrelation,
        )
    
    
    # Model
    config_train = load_yaml(args.config_train)
    if model_name in config_train:
        config_train = config_train[model_name]
    else:
        config_train = config_train['default']
        warnings.warn(f'No training configuration specified for {model_name}')
    print(f'config_train:\n{config_train}')
    
    model = load_model_by_mode(
        args=args, tokenizer=tokenizer, ntoken=ntoken)
    
    
    # # total_params = sum(p.numel() for p in model.parameters())
    
    if args.mode == 'training':
        # Train using HuggingFace Trainer
        train_with_trainer(args, model, tokenizer, dataset_dict, config_train,
                          qry_len, ans_len)
    
    elif args.mode == 'testing':
        mask_number = 5
        args.test_type = 'sft_abd'
        if args.merge_prob != 0.0:
            result = test(args, model, tokenizer, dataset_dict['test'], graph_samplers,mask_number, args.test_type)
        else:
            result = test_batch(args, model, tokenizer, dataset_dict['test'], graph_samplers, args.test_type)

    elif args.mode == 'optimizing':
        configs=load_yaml(args.config_rl)
        optimize(args, configs, dataset_dict, model, tokenizer)
        
   

if __name__ == '__main__':
    main()