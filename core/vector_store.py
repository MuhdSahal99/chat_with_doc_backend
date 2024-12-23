import faiss
import numpy as np
import pickle
import os
from typing import List

class VectorStore:
    def __init__(self, dimension: int, index_path: str):
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.texts = []
        
        # Create the directory structure when initializing
        os.makedirs(self.index_path, exist_ok=True)

    def create_index(self, embeddings: List[np.ndarray], texts: List[str]):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings))
        self.texts = texts
        self._save_index()

    def search(self, query_embedding: np.ndarray, k: int = 4) -> List[str]:
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k
        )
        return [self.texts[i] for i in indices[0]]

    def _save_index(self):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save the FAISS index
        index_file = os.path.join(self.index_path, "index.faiss")
        faiss.write_index(self.index, index_file)
        
        # Save the texts
        texts_file = os.path.join(self.index_path, "texts.pkl")
        with open(texts_file, "wb") as f:
            pickle.dump(self.texts, f)

    def load_index(self):
        index_file = os.path.join(self.index_path, "index.faiss")
        texts_file = os.path.join(self.index_path, "texts.pkl")
        
        self.index = faiss.read_index(index_file)
        with open(texts_file, "rb") as f:
            self.texts = pickle.load(f)