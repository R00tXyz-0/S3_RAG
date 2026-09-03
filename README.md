# RAG Ingestion & Chunking — Cours Bases de Données Avancées (Oracle / PL/SQL)



## RAG System

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines information retrieval with Large Language Models (LLMs).

Instead of asking an LLM to answer only from its internal knowledge, RAG first retrieves relevant information from a private knowledge base and then provides that information to the LLM as context.

This helps the model generate answers that are more relevant and grounded in the provided documents.

### What does this project do?

This project implements a RAG system that allows users to ask questions about a collection of documents.

The documents are processed and divided into smaller **chunks**. Each chunk is converted into a numerical representation called an **embedding**. These embeddings are stored in a vector database using **FAISS**.

When a user asks a question:

1. The question is converted into an embedding.
2. FAISS searches for the most relevant document chunks.
3. The retrieved chunks are provided as context to the **Gemini LLM**.
4. Gemini generates an answer based on the retrieved information.

The current system is designed to work with document content such as **PDFs**, with support for other document types and multimodal content planned as the project evolves.

### Why this is a RAG ingestion pipeline
Downstream RAG needs small, self-contained, well-described passages. Raw PDF pages are not
usable directly: ~31% of the Oracle PDF has no text layer, slides repeat titles, and footers
pollute the text. The pipeline normalises all of that into chunks ready for embedding later.


### Technologies

* Python
* Sentence Transformers
* FAISS
* Gemini API
* OCR / Document Processing

### Objective

The main objective of this project is to understand and implement a complete **Retrieval-Augmented Generation pipeline**, from document processing and semantic search to LLM-based answer generation.

```
