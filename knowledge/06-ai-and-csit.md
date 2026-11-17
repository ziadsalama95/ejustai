---
title: Artificial Intelligence, RAG & CSIT Concepts
url: https://www.ejust.edu.eg/csit
---

# Artificial Intelligence & RAG Knowledge Base

## Core AI Disciplines
- **Machine Learning (ML):** Supervised learning (classification, regression), unsupervised learning (clustering, dimensionality reduction), and reinforcement learning.
- **Deep Learning:** Neural networks, Convolutional Neural Networks (CNNs) for vision, Transformers and attention mechanisms for language and multimodal tasks.
- **Natural Language Processing (NLP):** Tokenization, embeddings, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and prompt engineering.

## How Retrieval-Augmented Generation (RAG) Works
1. **Document Ingestion & Chunking:** Source documents are split into overlapping semantic chunks so each chunk fits cleanly in an embedding context window.
2. **Vector Embeddings:** An embedding model (such as `nvidia/nv-embedqa-e5-v5`) converts each chunk into a dense numerical vector capturing semantic meaning.
3. **Vector Indexing:** Vectors and metadata are stored in a vector database (such as Cloudflare Vectorize) for fast cosine-similarity search.
4. **Retrieval & Grounded Generation:** When a user asks a question, the query is embedded, the top-K most relevant chunks are retrieved, and an LLM (such as Llama 3.3 70B on NVIDIA NIM) generates an accurate, source-cited answer grounded in those chunks.
