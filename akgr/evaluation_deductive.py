import torch
def evaluate_entailment(source_list, pred_list, nentity=24624):
    """
    修改后的函数，用于在ceshi.py中根据source和pred运行
    source: entailed_answers (ground truth)
    pred: query_answers (predicted answers)
    nentity: 实体总数，默认24624
    
    :param source_list: 列表，每个元素是字符串格式的实体ID，如 ['2746 3333 3452 5524 9919', '1269 3510 9822...']
    :param pred_list: 列表，每个元素是字符串格式的实体ID，如 ['2746 3333 3452 5524 9919', '1269 3510 9822...']
    :param nentity: 实体总数
    :return: 评估结果列表
    """
    
    from akgr.utils.parsing_util import unshift_entity_index
    
    log_list = []
    
    for i in range(len(source_list)):
        # 解析source (entailed_answers) 并使用unshift_entity_index处理
        source_str = source_list[i]
        entailed_answers = [unshift_entity_index(int(x)) for x in source_str.split()]
        
        # 解析pred (query_answers) 并使用unshift_entity_index处理
        pred_str = pred_list[i]
        query_answers = [unshift_entity_index(int(x)) for x in pred_str.split()]
        
        # 创建模拟的query_encoding (这里我们假设所有预测的实体都有相同的分数)
        # 在实际应用中，这个应该是模型输出的logits
        query_encoding = torch.zeros(1, nentity)
        
        # 给预测的实体分配分数
        for entity_id in query_answers:
            if 0 <= entity_id < nentity:
                query_encoding[0, entity_id] = 1.0
        
        # [batch_size, num_entities]
        all_scoring = query_encoding

        # [batch_size, num_entities]
        original_scores = all_scoring.clone()

        entailed_answer_set = torch.tensor(entailed_answers)

        # [num_entities]
        not_answer_scores = all_scoring[0]  # 取第一个batch
        not_answer_scores[entailed_answer_set] = - 10000000

        # [1, num_entities]
        not_answer_scores = not_answer_scores.unsqueeze(0)

        # [num_entailed_answers, 1]
        entailed_answers_scores = original_scores[0][entailed_answer_set].unsqueeze(1)

        # [num_entailed_answers, num_entities]
        answer_is_smaller_matrix = ((entailed_answers_scores - not_answer_scores) < 0)

        # [num_entailed_answers, num_entities]
        answer_is_equal_matrix = ((entailed_answers_scores - not_answer_scores) == 0)

        # [num_entailed_answer]
        answer_tied_num = answer_is_equal_matrix.sum(dim = -1)

        # [num_entailed_answer]
        random_tied_addition = torch.mul(torch.rand(answer_tied_num.size()).to(answer_tied_num.device), answer_tied_num).type(torch.int64)

        # [num_entailed_answers]
        answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

        # [num_entailed_answers]
        answer_rankings = torch.add(answer_rankings, random_tied_addition)

        # [num_entailed_answers]
        rankings = answer_rankings.float()

        mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
        hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
        hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
        hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()

        num_answers = len(entailed_answers)

        logs = {
            "ent_mrr": mrr,
            "ent_hit_at_1": hit_at_1,
            "ent_hit_at_3": hit_at_3,
            "ent_hit_at_10": hit_at_10,
            "ent_num_answers": num_answers
        }

        log_list.append(logs)
    
    # 计算平均指标
    if len(log_list) > 0:
        avg_mrr = sum(r['ent_mrr'] for r in log_list) / len(log_list)
        avg_hit1 = sum(r['ent_hit_at_1'] for r in log_list) / len(log_list)
        avg_hit3 = sum(r['ent_hit_at_3'] for r in log_list) / len(log_list)
        avg_hit10 = sum(r['ent_hit_at_10'] for r in log_list) / len(log_list)
        
        return {
            "avg_mrr": avg_mrr,
            "avg_hit_at_1": avg_hit1,
            "avg_hit_at_3": avg_hit3,
            "avg_hit_at_10": avg_hit10,
            "num_samples": len(log_list)
        }
    else:
        return {
            "avg_mrr": 0.0,
            "avg_hit_at_1": 0.0,
            "avg_hit_at_3": 0.0,
            "avg_hit_at_10": 0.0,
            "num_samples": 0
        }


def jaccard_similarity(set1, set2):
    """
    计算两个集合的Jaccard相似度
    Jaccard = |A ∩ B| / |A ∪ B|
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def calculate_average_jaccard(source_list, pred_list):
    """
    计算两个列表的平均Jaccard相似度
    """
    total_jaccard = 0
    count = 0
    
    for i in range(min(len(source_list), len(pred_list))):
        # 将字符串转换为整数集合
        source_set = set(map(int, source_list[i].split()))
        pred_set = set(map(int, pred_list[i].split()))
        
        # 计算Jaccard相似度
        jaccard = jaccard_similarity(source_set, pred_set)
        total_jaccard += jaccard
        count += 1
    
    average_jaccard = total_jaccard / count if count > 0 else 0
    return average_jaccard

def evaluate_deductive_batch(source_list, pred_list, nentity=24624):
    """
    批量评估函数，返回每个样本的详细信息（与abd格式对齐）
    
    :param source_list: 列表，每个元素是字符串格式的实体ID，如 ['2746 3333 3452 5524 9919', '1269 3510 9822...']
    :param pred_list: 列表，每个元素是字符串格式的实体ID，如 ['2746 3333 3452 5524 9919', '1269 3510 9822...']
    :param nentity: 实体总数，默认24624
    :return: 包含每个样本评估指标的列表
    """
    
    from akgr.utils.parsing_util import unshift_entity_index
    
    results = []
    
    for i in range(len(source_list)):
        # 解析source和pred
        source_str = source_list[i]
        pred_str = pred_list[i]
        
        source_set = set(map(int, source_str.split()))
        pred_set = set(map(int, pred_str.split()))
        
        # 计算Jaccard相似度
        jaccard = jaccard_similarity(source_set, pred_set)
        
        # 计算Entailment指标
        entailed_answers = [unshift_entity_index(int(x)) for x in source_str.split()]
        query_answers = [unshift_entity_index(int(x)) for x in pred_str.split()]
        
        # 创建模拟的query_encoding
        query_encoding = torch.zeros(1, nentity)
        for entity_id in query_answers:
            if 0 <= entity_id < nentity:
                query_encoding[0, entity_id] = 1.0
        
        all_scoring = query_encoding
        original_scores = all_scoring.clone()
        entailed_answer_set = torch.tensor(entailed_answers)
        
        not_answer_scores = all_scoring[0]
        not_answer_scores[entailed_answer_set] = -10000000
        not_answer_scores = not_answer_scores.unsqueeze(0)
        
        # 使用与原始evaluate_entailment相同的计算逻辑
        # [num_entailed_answers, 1]
        entailed_answers_scores = original_scores[0][entailed_answer_set].unsqueeze(1)
        
        # [num_entailed_answers, num_entities]
        answer_is_smaller_matrix = ((entailed_answers_scores - not_answer_scores) < 0)
        
        # [num_entailed_answers, num_entities]
        answer_is_equal_matrix = ((entailed_answers_scores - not_answer_scores) == 0)
        
        # [num_entailed_answer]
        answer_tied_num = answer_is_equal_matrix.sum(dim = -1)
        
        # [num_entailed_answer]
        random_tied_addition = torch.mul(torch.rand(answer_tied_num.size()).to(answer_tied_num.device), answer_tied_num).type(torch.int64)
        
        # [num_entailed_answers]
        answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1
        
        # [num_entailed_answers]
        answer_rankings = torch.add(answer_rankings, random_tied_addition)
        
        # [num_entailed_answers]
        rankings = answer_rankings.float()
        
        mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy().item()
        hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy().item()
        hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy().item()
        hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy().item()
        
        # 为每个样本创建一个字典
        sample_result = {
            'jaccard': jaccard,
            'mrr': mrr,
            'hit_at_1': hit_at_1,
            'hit_at_3': hit_at_3,
            'hit_at_10': hit_at_10
        }
        
        results.append(sample_result)
    
    return results

def evaluate_deductive(source_list, pred_list, nentity=24624):
    """
    综合评估函数，结合Jaccard相似度和Entailment指标
    
    :param source_list: 列表，每个元素是字符串格式的实体ID，如 ['2746 3333 3452 5524 9919', '1269 3510 9822...']
    :param pred_list: 列表，每个元素是字符串格式的实体ID，如 ['2746 3333 3452 5524 9919', '1269 3510 9822...']
    :param nentity: 实体总数，默认24624
    :return: 包含所有评估指标的字典
    """
    
    # 计算Jaccard相似度
    avg_jaccard = calculate_average_jaccard(source_list, pred_list)
    
    # 计算Entailment指标
    entailment_results = evaluate_entailment(source_list, pred_list, nentity)
    
    # 合并所有指标
    results = {
        "avg_jaccard": avg_jaccard,
        "avg_mrr": entailment_results["avg_mrr"],
        "avg_hit_at_1": entailment_results["avg_hit_at_1"],
        "avg_hit_at_3": entailment_results["avg_hit_at_3"],
        "avg_hit_at_10": entailment_results["avg_hit_at_10"],
        "num_samples": entailment_results["num_samples"]
    }
    
    return results

