import json
from pathlib import Path

import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add_embeddings(self, embeddings, documents):
        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_embedding, top_k=5):
        query = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            document = self.documents[index]

            results.append({
                "score": float(score),
                "chunk_id": document["chunk_id"],
                "text": document["text"],
                "metadata": document.get("metadata", {})
            })

        return results

    def save(self, index_path, documents_path):
        index_path = Path(index_path)
        documents_path = Path(documents_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with open(documents_path, "w", encoding="utf-8") as f:
            json.dump(
                self.documents,
                f,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, index_path, documents_path):

        index = faiss.read_index(str(index_path))

        store = cls(index.d)

        store.index = index

        with open(documents_path, "r", encoding="utf-8") as f:
            store.documents = json.load(f)

        return store