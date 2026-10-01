# 📚 RAG AI Study Assistant

An intelligent, context-aware AI Study Assistant built using **Retrieval-Augmented Generation (RAG)**. This application allows users to index Python programming notes (PDFs), convert them into vector embeddings, and query them in real-time to receive accurate, context-bound answers.

---

## 🌟 Key Features

- **Document Processing & Chunking**: Automatically reads PDF documents and splits text into manageable chunks using custom token/character overlap.
- **Vector Storage with ChromaDB**: Stores document embeddings locally in persistent storage for quick semantic similarity searches.
- **Fast Inferencing with Groq API**: Utilizes high-speed Large Language Models (LLMs) hosted on Groq to generate precise responses based *only* on the provided context.
- **Interactive UI**: Clean interface to process notes and ask questions seamlessly.
- **Serverless Ready**: Configured for deployment on Vercel with temporary directory handling for ChromaDB.

---

## 🛠️ Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **LLM Provider**: [Groq API](https://groq.com/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **PDF Extraction**: `pypdf`
- **Frontend**: HTML5, CSS3, JavaScript

---

## 📂 Project Structure

```text
rag-study-assistant/
├── db/                  # Persistent ChromaDB vector storage (Local)
├── pdfs/                # Directory for source PDF documents
├── static/              # CSS and static assets
│   └── style.css
├── .env                 # Environment variables (API Keys)
├── .gitignore           # Git ignore file
├── index.html           # Main frontend user interface
├── main.py              # FastAPI application & RAG logic
├── requirements.txt     # Python dependencies
└── vercel.json          # Deployment configuration for Vercel
