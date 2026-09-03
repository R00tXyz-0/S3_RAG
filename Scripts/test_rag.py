import os
from dotenv import load_dotenv

from src.RAG.pipeline import RAGPipeline


load_dotenv()


rag = RAGPipeline(
    index_path="data/processed/embeddings/oracle.index",
    documents_path="data/processed/embeddings/oracle_documents.json",
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


query = "Création du dictionnaire de données"

result = rag.ask(query, top_k=5)

print("\n" + "=" * 70)
print("REPONSE")
print("=" * 70)

print(result["answer"])

print("\n" + "=" * 70)
print("SOURCES")
print("=" * 70)

for source in result["sources"]:
    print(
        source["chunk_id"],
        source["score"]
    )