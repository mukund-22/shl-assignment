import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SHLRecommender:
    def __init__(self, catalog_path="data/processed/assessments_clean.csv", model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=5):
        self.top_k = top_k
        self.catalog_path = catalog_path
        self.model = SentenceTransformer(model_name)
        self.catalog = self.load_catalog()
        self.catalog_embeddings = self.embed_catalog()

    def load_catalog(self):
        """Loads the catalog from a CSV file into a pandas DataFrame."""
        return pd.read_csv(self.catalog_path)

    def embed_catalog(self):
        """Embeds the catalog descriptions using the SentenceTransformer model."""
        descriptions = self.catalog["combined_text"].tolist()
        return self.model.encode(descriptions, show_progress_bar=False)

    def recommend(self, user_input):
        """Recommends assessments based on the user input."""
        input_embedding = self.model.encode([user_input])
        similarities = cosine_similarity(input_embedding, self.catalog_embeddings)[0]
        top_indices = similarities.argsort()[::-1][:self.top_k]
        recommendations = self.catalog.iloc[top_indices]
        return recommendations.to_dict(orient="records")
