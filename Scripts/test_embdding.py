from src.embeddings import build_embeddings


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

build_embeddings(
    chunks_path="data/processed/chunks/chunks_Cours_Oracle_.jsonl",
    output_path="data/processed/embeddings/Oracle_embeddings.jsonl",
    model_name=MODEL_NAME
)

