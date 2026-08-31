from src.vector_store.faiss_store import FAISSVectorStore
from src.embeddings.embddingM import EmbeddingModel


class Retriever:

    def __init__(
        self,
        index_path: str,
        documents_path: str,
        model_name: str
    ):
        self.store = FAISSVectorStore.load(
            index_path,
            documents_path
        )

        self.embedding_model = EmbeddingModel(
            model_name
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):
        query_embedding = self.embedding_model.encode_query(
            query
        )

        results = self.store.search(
            query_embedding,
            top_k
        )

        return results