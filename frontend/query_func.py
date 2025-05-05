import pandas as pd
from backend.recommender_llm import SHLRecommenderLLM

recommender = SHLRecommenderLLM()

def query_handling_using_LLM_updated(query: str) -> pd.DataFrame:
    results = recommender.recommend(query=query, top_k=10)
    if not results:
        return pd.DataFrame()
    # Convert results to DataFrame with expected columns
    df = pd.DataFrame(results)
    # Rename columns to match frontend expectations
    df = df.rename(columns={
        "name": "Assessment Name",
        "skills": "Skills",
        "test_type": "Test Type",
        "combined_text": "Description",
        "remote_support": "Remote Testing Support",
        "adaptive_support": "Adaptive/IRT",
        "duration": "Duration",
        "url": "URL"
    })
    return df
