import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Groq API key from environment variable

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = os.environ.get("GROQ_API_URL")
if not GROQ_API_URL:
    GROQ_API_URL = "https://api.groq.com/openai/v1/embeddings"

if not GROQ_API_KEY:
    raise ValueError("Environment variable GROQ_API_KEY is not set. Please set it before running the application.")

import json

class SHLRecommenderLLM:
    def __init__(self, data_path='data/processed/assessments_clean.csv', test_size=0.3, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.df = pd.read_csv(self.data_path)
        
        # Split data into train and test sets
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )
        # Reset index
        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)
        
        logger.info(f"Recommender initialized with {len(self.train_df)} training assessments and {len(self.test_df)} test assessments.")

        # Prepare candidate assessments text for prompt
        self.candidate_texts = (self.train_df['name'].fillna('') + ' - ' +
                                self.train_df['combined_text'].fillna('') + ' - ' +
                                self.train_df['test_type'].fillna('').astype(str)).tolist()

        # Generate embeddings for candidate_texts for fallback
        self.candidate_embeddings = self._generate_embeddings(self.candidate_texts)

    def _generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts using Groq embeddings API.
        """
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-embedding-3-large",
            "input": texts
        }
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings = [item['embedding'] for item in data['data']]
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.zeros((len(texts), 768))  # fallback to zero embeddings if error

    def _embedding_fallback(self, query, top_k=10):
        """
        Fallback recommendation using cosine similarity of embeddings.
        """
        query_embedding = self._generate_embeddings([query])
        if query_embedding.shape[0] == 0:
            logger.error("Failed to generate embedding for query in fallback.")
            return []

        similarities = cosine_similarity(query_embedding, self.candidate_embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.train_df.iloc[idx].to_dict()
            row['rank'] = rank
            results.append(row)
        return results

class RateLimitExceededError(Exception):
    def __init__(self, retry_after_seconds):
        super().__init__(f"Rate limit exceeded, retry after {retry_after_seconds} seconds")
        self.retry_after_seconds = retry_after_seconds

def _call_groq_chat(self, messages):
    import time
    import re
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages
    }
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 or response.status_code == 524:
                # Try to parse retry time from response text
                retry_seconds = 60  # default 1 minute
                try:
                    match = re.search(r"Please try again in (\d+)m(\d+\.?\d*)s", response.text)
                    if match:
                        minutes = int(match.group(1))
                        seconds = float(match.group(2))
                        retry_seconds = minutes * 60 + seconds
                except Exception:
                    pass
                if retry_seconds > 300:  # more than 5 minutes
                    logger.error(f"Rate limit exceeded, retry time too long ({retry_seconds}s). Aborting.")
                    raise RateLimitExceededError(retry_seconds)
                logger.warning(f"Rate limit or timeout error during chat completion request: {e} - Response content: {response.text}. Retrying in {retry_seconds} seconds...")
                time.sleep(retry_seconds)
            else:
                logger.error(f"HTTP error during chat completion request: {e} - Response content: {response.text}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error during chat completion request: {e}")
            raise
    raise Exception("Max retries exceeded for Groq chat completion request")

def recommend(self, query, max_duration=None, top_k=10):
    import re
    if not hasattr(self, '_recommend_cache'):
        self._recommend_cache = {}

    if query in self._recommend_cache:
        return self._recommend_cache[query]

    batch_size = 20
    recommended_set = set()
    results = []
    llm_failed = True

    system_message = {
        "role": "system",
        "content": "You are an expert recommender system. Given a user query and a list of assessments, recommend the most relevant assessments."
    }

    for i in range(0, len(self.candidate_texts), batch_size):
        batch_texts = self.candidate_texts[i:i+batch_size]
        user_message = {
            "role": "user",
            "content": f"User query: {query}\n\nAssessments:\n" + "\n".join(batch_texts)
        }
        messages = [system_message, user_message]

        try:
            response_text = self._call_groq_chat(messages)
            llm_failed = False
        except Exception as e:
            # Log and continue with next batch
            logger.error(f"Error during Groq chat call for batch starting at {i}: {e}")
            continue

        # Parse response to extract recommended assessments
        for line in response_text.splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                rec = line.lstrip("0123456789. -").strip()
                if rec and rec not in recommended_set:
                    recommended_set.add(rec)
                    # Map recommended name back to assessments in train_df
                    pattern = re.escape(rec)
                    matches = self.train_df[self.train_df['name'].str.contains(pattern, case=False, na=False)]
                    if not matches.empty:
                        for _, row in matches.iterrows():
                            results.append(row.to_dict())
                            if len(results) >= top_k:
                                break
            if len(results) >= top_k:
                break
        if len(results) >= top_k:
            break

    if llm_failed:
        logger.info("LLM failed for all batches, falling back to embedding-based recommendation.")
        results = self._embedding_fallback(query, top_k=top_k)

    # Add rank
    for i, rec in enumerate(results):
        rec['rank'] = i + 1

    self._recommend_cache[query] = results
    return results
