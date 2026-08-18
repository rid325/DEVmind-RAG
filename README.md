# DEVmind RAG

An advanced, multi-source Retrieval-Augmented Generation (RAG) backend API built with FastAPI, PostgreSQL (pgvector), and Sentence Transformers. 

This project aims to consolidate and search technical knowledge across multiple domains, offering highly relevant results by combining dense semantic search, sparse keyword matching, and neural reranking.

## Features

- **Multi-Source Ingestion**: Automatically ingests and indexes documents from Arxiv, StackOverflow, and GitHub into a unified database.
- **Hybrid Search**: Combines Dense Vector Search (semantic similarity using embeddings) and Sparse Search (BM25 keyword matching) using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Re-evaluates and accurately re-orders the top candidates from the hybrid search using a specialized `ms-marco-MiniLM-L-6-v2` cross-encoder to guarantee the most contextually relevant results surface to the top.
- **FastAPI Backend**: Fast, asynchronous endpoints to trigger background ingestion jobs, embedding generations, and retrieval queries.

## Architecture & Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with `pgvector` extension for vector similarity search
- **ORM**: SQLAlchemy
- **Embeddings & ML Models**:
  - `sentence-transformers` for generating document embeddings
  - `CrossEncoder` for neural reranking
- **Sparse Search**: `rank_bm25`

## Endpoints

- `POST /ingest/arxiv`: Trigger background ingestion of Arxiv papers.
- `POST /ingest/stackoverflow`: Trigger background ingestion of StackOverflow data.
- `POST /ingest/github`: Trigger background ingestion of GitHub documents.
- `POST /embed`: Generate embeddings for newly ingested documents in the background.
- `GET /search/bm25`: Perform standard sparse keyword search.
- `GET /search/hybrid`: Perform an advanced hybrid search (Dense + Sparse).
  - Query Param `query`: The search string.
  - Query Param `k`: Number of results to return (default: 5).
  - Query Param `rerank`: Boolean to enable/disable Cross-Encoder reranking (default: `True`).
- `GET /stats`: View document counts across domains and embedding statuses.

## Getting Started

1. Clone the repository and install dependencies from `requirements.txt`.
2. Ensure you have a running instance of PostgreSQL with the `pgvector` extension enabled. (Refer to `docker-compose.yml` if applicable).
3. Set your database credentials in your `.env` file.
4. Run the server:
   ```bash
   uvicorn app.main:app --reload --port 8001
   ```

   daily check
   
