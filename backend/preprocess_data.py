import json
import pandas as pd
from pathlib import Path

def preprocess_data():
    # Load scraped data
    with open('data/raw/assessments.json') as f:
        assessments = json.load(f)
    # Convert to DataFrame
    df = pd.DataFrame(assessments)
    # Clean data
    df['combined_text'] = df.apply(lambda x: f"{x['name']} {x['test_type']}", axis=1)
    # Save processed data
    Path('data/processed').mkdir(exist_ok=True)
    df.to_csv('data/processed/assessments_clean.csv', index=False)
    return df

if __name__ == "__main__":
    preprocess_data()
