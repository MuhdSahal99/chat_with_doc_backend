from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class DocumentEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        return self.model.encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]