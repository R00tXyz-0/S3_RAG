import json
import numpy as np

from src.vector_store.faiss_store import FAISSVectorStore


EMBEDDINGS_PATH = (
    "data/processed/embeddings/Oracle_embeddings.jsonl" # OR ORACLE
)

INDEX_PATH = (
    "data/processed/embeddings/oracle.index"
)

DOCUMENTS_PATH = (
    "data/processed/embeddings/oracle_documents.json"
)

def load_embeddings(path):

    embeddings = []
    documents = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            item = json.loads(line)

            embedding = item["embedding"]

            document = {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "metadata": item.get("metadata", {})
            }

            embeddings.append(embedding)
            documents.append(document)

    return np.asarray(embeddings, dtype="float32"), documents


def main():

    print("Loading embeddings...")

    embeddings, documents = load_embeddings(
        EMBEDDINGS_PATH
    )

    print(f"Chunks: {len(documents)}")
    print(f"Embedding shape: {embeddings.shape}")

    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    store = FAISSVectorStore(dimension)

    store.add_embeddings(
        embeddings,
        documents
    )

    store.save(
        INDEX_PATH,
        DOCUMENTS_PATH
    )

    print("\nVector database created successfully!")
    print(f"Index: {INDEX_PATH}")
    print(f"Documents: {DOCUMENTS_PATH}")
    print(f"Vectors: {store.index.ntotal}")


if __name__ == "__main__":
    main()