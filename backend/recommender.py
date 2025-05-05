import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SHLRecommender:
    def __init__(self, catalog_path="data/catalog.json", model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=5):
        self.top_k = top_k
        self.catalog_path = catalog_path
        self.model = SentenceTransformer(model_name)  # Using PyTorch-based SentenceTransformer
        self.catalog = self.load_catalog()
        self.catalog_embeddings = self.embed_catalog()

    def load_catalog(self):
        """Loads the catalog from the JSON file."""
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def embed_catalog(self):
        """Embeds the catalog descriptions using the SentenceTransformer model."""
        descriptions = [item["description"] for item in self.catalog]
        return self.model.encode(descriptions, show_progress_bar=False)

    def recommend(self, user_input):
        """Recommends assessments based on the user input."""
        input_embedding = self.model.encode([user_input])
        similarities = cosine_similarity(input_embedding, self.catalog_embeddings)[0]
        top_indices = similarities.argsort()[::-1][:self.top_k]
        recommendations = [self.catalog[i] for i in top_indices]
        return recommendations
