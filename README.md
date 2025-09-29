# RAG (Retrieval-Augmented Generation) Demo using Ollama, FAISS, Langchain, and FastAPI

> **A hands-on project demonstrating how to combine document retrieval (FAISS) and text generation (Ollama LLM) into a simple but powerful API using FastAPI.**

---

## 🚀 Overview

This project shows how to build a **Retrieval-Augmented Generation (RAG)** system:
- **Retrieve** relevant chunks from your own document collection using a vector database (**FAISS**).
- **Augment** user queries with retrieved context.
- **Generate** accurate, context-aware responses using the **qwen3:0.6b** LLM running on **Ollama**.
- **Expose** all of this through a simple, production-ready **FastAPI** server.

## 🏗️ Architecture
User Query
│
▼
[FastAPI REST API]
│
▼
Retrieve Relevant Chunks
(FAISS Vector Search)
│
▼
Augment User Query
(add retrieved context)
│
▼
Generate Answer
(Ollama LLM - qwen3:0.6b)
│
▼
Return Response

Text_Generation/
│
├── main_ollama.py # FastAPI application (serves the RAG endpoint)
├── preprocess.py # Script: splits documents, generates embeddings, builds FAISS index
├── requirements.txt # All Python dependencies
├── venv/ # Python virtual environment (ignored by git)
├── pycache/ # Python bytecode cache (ignored by git)
└── documents/ # Directory containing source .txt/.pdf files

The API will:

Retrieve relevant document chunks.

Augment your prompt with the retrieved context.

Generate a coherent, context-aware answer using the Ollama LLM.

🧩 Key Components

Ollama LLM (qwen3:0.6b): Local, efficient large language model for text generation.

FAISS: Fast, lightweight vector database for nearest-neighbor retrieval.

Langchain: Powerful library for document loading and chunking.

FastAPI: Modern, async web framework for building APIs.

📑 Files Explained

preprocess.py:
Splits all your documents into chunks and creates the FAISS index with embeddings.

main_ollama.py:
FastAPI app. For each query, retrieves context using FAISS, then generates output using Ollama.

requirements.txt:
List of all dependencies.

❓ Troubleshooting

Import errors?

Double-check you are in the correct Python environment.

Run pip install -r requirements.txt again.

Restart VS Code after (re)installing libraries.

Ollama not running?

Start your Ollama server: ollama serve

Make sure the model (qwen3:0.6b) is available locally.

Can't find FAISS index?

Re-run python preprocess.py after adding new documents.

📄 License

This project is released under the MIT License

🙏 Credits

Langchain

Ollama

FAISS

FastAPI

