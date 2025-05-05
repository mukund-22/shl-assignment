import numpy as np
import pandas as pd

def recall_at_k(y_true, y_pred, k):
    """
    Compute recall at k for a single user/query.
    
    Args:
        y_true (list or set): Ground truth relevant items
        y_pred (list): Predicted items ordered by relevance
    
    Returns:
        float: recall@k value
    """
    y_true_set = set(y_true)
    y_pred_k = y_pred[:k]
    hits = sum(1 for item in y_pred_k if item in y_true_set)
    return hits / len(y_true_set) if y_true_set else 0.0

def mean_recall_at_k(all_y_true, all_y_pred, k):
    """
    Compute mean recall at k over multiple users/queries.
    
    Args:
        all_y_true (list of lists or sets): List of ground truth relevant items per user/query
        all_y_pred (list of lists): List of predicted items per user/query
    
    Returns:
        float: mean recall@k value
    """
    recalls = [recall_at_k(y_true, y_pred, k) for y_true, y_pred in zip(all_y_true, all_y_pred)]
    return np.mean(recalls)

def average_precision_at_k(y_true, y_pred, k):
    """
    Compute average precision at k for a single user/query.
    
    Args:
        y_true (list or set): Ground truth relevant items
        y_pred (list): Predicted items ordered by relevance
    
    Returns:
        float: average precision@k value
    """
    y_true_set = set(y_true)
    y_pred_k = y_pred[:k]
    hits = 0
    sum_precisions = 0.0
    for i, p in enumerate(y_pred_k, start=1):
        if p in y_true_set:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(y_true_set), k) if y_true_set else 0.0

def mean_average_precision_at_k(all_y_true, all_y_pred, k):
    """
    Compute mean average precision at k over multiple users/queries.
    
    Args:
        all_y_true (list of lists or sets): List of ground truth relevant items per user/query
        all_y_pred (list of lists): List of predicted items per user/query
    
    Returns:
        float: mean average precision@k value
    """
    average_precisions = [average_precision_at_k(y_true, y_pred, k) for y_true, y_pred in zip(all_y_true, all_y_pred)]
    return np.mean(average_precisions)

def compute_metrics(benchmark_queries, recommender, k=5):
    """
    Compute recall@k and MAP@k for benchmark queries using the provided recommender instance.
    """
    recall_scores = []
    average_precisions = []

    for entry in benchmark_queries:
        query = entry["query"]
        relevant_items = entry["relevant"]

        results = recommender.recommend(query, top_k=k)
        topk = [res["name"] for res in results[:k]]

        # recall@k
        count = 0
        for item in topk:
            if item in relevant_items:
                count += 1
        recall_score = count / len(relevant_items) if relevant_items else 0.0
        recall_scores.append(min(recall_score, 1.0))

        # map@k
        ap = 0.0
        relevant_count = 0
        for i, res in enumerate(topk):
            if res in relevant_items:
                relevant_count += 1
                precision_at_k = relevant_count / (i + 1)
                ap += precision_at_k
        ap = ap / min(k, len(relevant_items)) if relevant_items else 0.0
        average_precisions.append(min(ap, 1.0))

    recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    map_ = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0

    print(f"Recall@{k}: {recall:.4f}")
    print(f"MAP@{k}: {map_:.4f}")
