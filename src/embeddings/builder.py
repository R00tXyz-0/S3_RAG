import json
import numpy as np

from pathlib import Path
from .embddingM import EmbeddingModel


def build_embeddings(
    chunks_path: str,
    output_path: str,
    model_name: str
):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    texts = [chunk["text"] for chunk in chunks]

    model = EmbeddingModel(model_name)

    embeddings = model.encode(texts)

    output = []

    for chunk, embedding in zip(chunks, embeddings):
        output.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "embedding": embedding.tolist()
        })

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")