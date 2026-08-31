from src.retrieval.retriever import Retriever
from src.LLM.gemini import GeminiLLM


class RAGPipeline:

    def __init__(
        self,
        index_path,
        documents_path,
        embedding_model,
        gemini_model="gemini-3.7-flash"
    ):

        self.retriever = Retriever(
            index_path=index_path,
            documents_path=documents_path,
            model_name=embedding_model
        )

        self.llm = GeminiLLM(
            model_name=gemini_model
        )

    def ask(self, query: str, top_k: int = 5):

        results = self.retriever.retrieve(
            query,
            top_k=top_k
        )

        context = "\n\n".join(
            [
                f"[Source: {r['chunk_id']}]\n{r['text']}"
                for r in results
            ]
        )

        prompt = f"""
Tu es un assistant spécialisé dans les cours universitaires.

Réponds uniquement à partir du contexte fourni.
Si l'information n'est pas présente dans le contexte,
dis clairement que l'information n'est pas disponible.

Contexte:
{context}

Question:
{query}

Réponse:
"""

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": results
        }