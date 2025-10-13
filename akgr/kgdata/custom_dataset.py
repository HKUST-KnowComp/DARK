import torch
from torch.utils.data import Dataset

class AbductionDataset(Dataset):
    def __init__(self, dataset_dict, tokenizer, query_len, answer_len, training_mode):
        self.dataset = dataset_dict
        self.tokenizer = tokenizer
        self.answer_len = answer_len + 1
        self.query_len = query_len + 1
        self.max_len = answer_len + query_len + 2
        self.training_mode = training_mode
        self.max_delete = self.max_len
        
    def __len__(self):
        return len(self.dataset)
    
    def unify_extract_sample_to_device(self, sample, model_type='unify'):
        """Extract and process sample data for the model"""
        observation = sample['source']
        query = sample['target']
        pattern_id = sample['pattern_id']
        
        # Create the custom format:  [bos] + query + [SEP] + observation + [eos] + padding
        
        # Process single observation and query (not batch)
        
        query_tokenized = self.tokenizer(
            query,
            add_special_tokens=False,
            return_tensors="pt")
        
        observation_tokenized = self.tokenizer(
            observation,
            add_special_tokens=False,
            return_tensors="pt")
        
        observation_ids = observation_tokenized["input_ids"]  # Shape: [1, seq_len]
        query_ids = query_tokenized["input_ids"]  # Shape: [1, seq_len]

        observation_length = observation_ids.shape[1]
        query_length = query_ids.shape[1]

        # For single sample, batch_size = 1
        sep_id = torch.full((1, 1), self.tokenizer.sep_token_id)
        bos_id = torch.full((1, 1), self.tokenizer.bos_token_id)
        eos_id = torch.full((1, 1), self.tokenizer.eos_token_id)
        
        
        
        if model_type == 'unify':

            if self.max_len -query_length-observation_length-2 > 0 and self.max_delete > 0:
                eos_count = torch.randint(
                    low=0,
                    high=min(self.max_delete, self.max_len -query_length-observation_length-2),
                    size=(1,),
                ).item()
                eos_tensor = torch.full((1, eos_count), self.tokenizer.eos_token_id, dtype=query_ids.dtype)
            else:
                eos_count = 0
                eos_tensor = torch.empty((1, 0), dtype=query_ids.dtype)

            # Concatenate query + sep + observation
            # Concatenate query + sep + observation
            middle_ids = torch.cat([query_ids, sep_id, observation_ids, eos_tensor], dim=1)
            input_ids = torch.cat([bos_id, middle_ids, eos_id], dim=1)
            attention_mask = torch.cat([torch.zeros((1, 1)), torch.ones_like(middle_ids), torch.zeros((1, 1))], dim=1)
            
        elif model_type == 'sft_abd':
            if self.query_len - query_length - 1> 0 and self.max_delete > 0:
                eos_count = torch.randint(
                    low=0,
                    high=min(self.max_delete, self.query_len - query_length - 1),
                    size=(1,),
                ).item()
                eos_tensor_query = torch.full((1, eos_count), self.tokenizer.eos_token_id, dtype=query_ids.dtype)
            
            else:
                eos_count = 0
                eos_tensor_query = torch.empty((1, 0), dtype=query_ids.dtype)

            middle_ids = torch.cat([query_ids, eos_tensor_query], dim=1)
            end_ids = torch.cat([sep_id, observation_ids, eos_id], dim=1)
            input_ids = torch.cat([bos_id, middle_ids, end_ids], dim=1)
            attention_mask = torch.cat([torch.zeros_like(bos_id), torch.ones_like(middle_ids), torch.zeros_like(end_ids)], dim=1)
            
        elif model_type == 'sft_ded':
            if self.answer_len - observation_length - 1 > 0 and self.max_delete > 0:
                eos_count = torch.randint(
                low=0,
                high=min(self.max_delete, self.answer_len - observation_length - 1),
                size=(1,),
                ).item()
                eos_tensor_observation = torch.full((1, eos_count), self.tokenizer.eos_token_id, dtype=query_ids.dtype)
            else:
                eos_count = 0
                eos_tensor_observation = torch.empty((1, 0), dtype=query_ids.dtype)

            middle_ids = torch.cat([observation_ids, eos_tensor_observation], dim=1)
            bos_ids = torch.cat([bos_id, query_ids, sep_id], dim=1)
            input_ids = torch.cat([bos_ids, middle_ids, eos_id], dim=1)
            attention_mask = torch.cat([torch.zeros_like(bos_ids), torch.ones_like(middle_ids), torch.zeros_like(eos_id)], dim=1)

        
        # Calculate current length and padding needed
        current_length = input_ids.shape[1]
        max_length = self.max_len
        
        if current_length < max_length:
            # Calculate padding length
            padding_length = max_length - current_length
            
            # Create padding tokens using eos_token_id
            padding_ids = torch.full((1, padding_length), self.tokenizer.eos_token_id)
            padding_mask = torch.zeros((1, padding_length))
            
            # Concatenate padding to the right
            input_ids = torch.cat([input_ids, padding_ids], dim=1)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=1)
        
        return observation, query, pattern_id, input_ids, attention_mask
            
    def __getitem__(self, idx):
        sample = self.dataset[idx]  
        # Extract sample data using the class method
        if self.training_mode == 'unify':
            observation, query, pattern_id, input_ids, attention_mask = \
                self.unify_extract_sample_to_device(sample, model_type='unify')
        elif self.training_mode == 'sft':
            # 随机选择 sft_ded 或 sft_adb
            if torch.rand(1).item() < 0.5:
                # sft_deductive 模式
                observation, query, pattern_id, input_ids, attention_mask = \
                    self.unify_extract_sample_to_device(sample, model_type='sft_ded')
            else:
                # sft_abductive 模式
                observation, query, pattern_id, input_ids, attention_mask = \
                    self.unify_extract_sample_to_device(sample, model_type='sft_abd')
        
        # Convert to single samples (remove batch dimension)
        result = {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0)
        }
            
        return result
        
#do not change the length
class AbductionDataset_length(Dataset):
    def __init__(self, dataset_dict, tokenizer, query_len, answer_len, training_mode):
        self.dataset = dataset_dict
        self.tokenizer = tokenizer
        self.answer_len = answer_len + 1
        self.query_len = query_len + 1
        self.max_len = answer_len + query_len + 2
        self.training_mode = training_mode
        self.max_delete = self.max_len
        
    def __len__(self):
        return len(self.dataset)
    
    def unify_extract_sample_to_device(self, sample, model_type='unify'):
        """Extract and process sample data for the model"""
        observation = sample['source']
        query = sample['target']
        pattern_id = sample['pattern_id']
        
        # Create the custom format:  [bos] + query + [SEP] + observation + [eos] + padding
        
        # Process single observation and query (not batch)
        
        query_tokenized = self.tokenizer(
            query,
            add_special_tokens=False,
            max_length=self.query_len,
            padding='max_length',
            return_tensors="pt")
        
        observation_tokenized = self.tokenizer(
            observation,
            add_special_tokens=False,
            max_length=self.answer_len,
            padding='max_length',
            return_tensors="pt")
        
        observation_ids = observation_tokenized["input_ids"]  # Shape: [1, seq_len]
        query_ids = query_tokenized["input_ids"]  # Shape: [1, seq_len]

        observation_length = observation_ids.shape[1]
        query_length = query_ids.shape[1]

        # For single sample, batch_size = 1
        sep_id = torch.full((1, 1), self.tokenizer.sep_token_id)
        bos_id = torch.full((1, 1), self.tokenizer.bos_token_id)
        eos_id = torch.full((1, 1), self.tokenizer.eos_token_id)
        
        
                  
        input_ids = torch.cat([bos_id, query_ids, sep_id, observation_ids, eos_id], dim=1)
        if model_type == 'unify':
            attention_mask = torch.cat([torch.zeros((1, 1)), torch.ones_like(query_ids), torch.zeros((1, 1)), torch.ones_like(observation_ids), torch.zeros((1, 1))], dim=1)
            
        elif model_type == 'sft_abd':
            attention_mask = torch.cat([torch.zeros_like(bos_id), torch.ones_like(query_ids), torch.zeros((1, 1)), torch.zeros_like(observation_ids), torch.zeros((1, 1))], dim=1)
            
        elif model_type == 'sft_ded':
            attention_mask = torch.cat([torch.zeros_like(bos_id), torch.zeros_like(query_ids), torch.zeros((1, 1)), torch.ones_like(observation_ids), torch.zeros((1, 1))], dim=1)

        
        
        return observation, query, pattern_id, input_ids, attention_mask
            
    def __getitem__(self, idx):
        sample = self.dataset[idx]  
        # Extract sample data using the class method
        if self.training_mode == 'unify':
            observation, query, pattern_id, input_ids, attention_mask = \
                self.unify_extract_sample_to_device(sample, model_type='unify')
        elif self.training_mode == 'sft':
            # 随机选择 sft_ded 或 sft_adb
            if torch.rand(1).item() < 0.5:
                # sft_deductive 模式
                observation, query, pattern_id, input_ids, attention_mask = \
                    self.unify_extract_sample_to_device(sample, model_type='sft_ded')
            else:
                # sft_abductive 模式
                observation, query, pattern_id, input_ids, attention_mask = \
                    self.unify_extract_sample_to_device(sample, model_type='sft_abd')
        
        # Convert to single samples (remove batch dimension)
        result = {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0)
        }
            
        return result

class GRPODataset(Dataset):
    def __init__(self, dataset_dict):
        self.dataset = dataset_dict['train']  # 直接使用train数据集
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]  
        
        # Check if sample is None or missing required keys
        if sample is None:
            raise ValueError(f"Sample at index {idx} is None")
        
        if not isinstance(sample, dict):
            raise ValueError(f"Sample at index {idx} is not a dictionary: {type(sample)}")
            
        required_keys = ['source', 'target', 'pattern_id']
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise ValueError(f"Sample at index {idx} missing keys: {missing_keys}")
        
        # Convert to single samples (remove batch dimension)
        result = {
            'source': sample['source'],
            'target': sample['target'],
            'pattern_id': sample['pattern_id']
        }
            
        return result