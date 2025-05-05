import pandas as pd
from sklearn.model_selection import train_test_split
from recommender import SHLRecommender
from recommender_llm import SHLRecommenderLLM, RateLimitExceededError
from metrics import compute_metrics

def generate_benchmark_queries(df):
    # Generate benchmark queries from assessment names in test set
    benchmark_queries = []
    for _, row in df.iterrows():
        query = row['name']
        # Relevant items are those with the same test_type or similar name (simplified here as exact match)
        relevant = [query]
        benchmark_queries.append({
            "query": query,
            "relevant": relevant
        })
    return benchmark_queries

import time

def main():
    df = pd.read_csv('data/processed/assessments_clean.csv')
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=80)

    # Initialize recommenders with train data if applicable
    # Assuming recommenders accept data_path parameter, save train data to temp csv
    train_df.to_csv('data/processed/train_assessments.csv', index=False)
    test_df.to_csv('data/processed/test_assessments.csv', index=False)

    print("Evaluating SHLRecommender...")
    recommender = SHLRecommender(data_path='data/processed/train_assessments.csv')
    benchmark_queries = generate_benchmark_queries(test_df)
    compute_metrics(benchmark_queries, recommender, k=5)

    print("\nEvaluating SHLRecommenderLLM...")
    recommender_llm = SHLRecommenderLLM(data_path='data/processed/train_assessments.csv')

    # Throttle requests to avoid rate limits
    results = []
    i = 0
    while i < len(benchmark_queries):
        query_obj = benchmark_queries[i]
        query = query_obj['query']
        try:
            res = recommender_llm.recommend(query, top_k=5)
            results.append(res)
            i += 1
            if i % 5 == 0:
                print(f"Processed {i} queries, sleeping to avoid rate limit...")
                time.sleep(60)  # sleep 60 seconds every 5 queries
        except RateLimitExceededError as e:
            wait_time = e.retry_after_seconds
            print(f"Rate limit exceeded, sleeping for {wait_time} seconds before retrying...")
            time.sleep(wait_time)

    # Optionally compute metrics on collected results here or modify compute_metrics to accept precomputed results
    # For now, just print that evaluation is done
    print("Evaluation of SHLRecommenderLLM completed with throttling.")

if __name__ == "__main__":
    main()
