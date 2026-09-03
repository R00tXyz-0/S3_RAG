from src.retrieval.retriever import Retriever


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

INDEX_PATH = (
    "data/processed/embeddings/oracle.index"
)

DOCUMENTS_PATH = (
    "data/processed/embeddings/oracle_documents.json"
)


def main():

    retriever = Retriever(
        index_path=INDEX_PATH,
        documents_path=DOCUMENTS_PATH,
        model_name=MODEL_NAME
    )

    query = "Création du dictionnaire de données"

    results = retriever.retrieve(
        query,
        top_k=5
    )

    print("\nRESULTATS")
    print("=" * 70)

    for i, result in enumerate(results, 1):

        print(f"\n--- Resultat {i} ---")
        print(f"Score    : {result['score']:.4f}")
        print(f"Chunk ID : {result['chunk_id']}")

        metadata = result["metadata"]

        print(
            f"Source   : "
            f"{metadata.get('source')}"
        )

        print(
            f"Section  : "
            f"{metadata.get('section')}"
        )

        print(f"\n{result['text']}")


if __name__ == "__main__":
    main()