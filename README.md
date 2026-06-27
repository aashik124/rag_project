# Palm Mind AI RAG Backend

FastAPI backend for document ingestion and conversational RAG with interview booking.

## Features

- Upload `.pdf` or `.txt` documents
- Extract document text
- Selectable chunking strategies:
  - `recursive`: separator-aware overlapping chunks
  - `semantic`: paragraph/sentence grouped chunks
- Generate embeddings with a local sentence-transformers model
- Store vectors in Qdrant or Pinecone
- Store document, chunk, and booking metadata in SQL database
- Custom conversational RAG implementation without `RetrievalQAChain`
- Redis-backed multi-turn chat memory
- LLM-assisted interview booking extraction

## Tech Stack

- FastAPI
- SQLAlchemy 2.0
- PostgreSQL or SQLite
- Redis
- Qdrant or Pinecone
- sentence-transformers embeddings
- Gemini chat completions


## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` with your gemini API keys and service URLs.

## Run

```bash
uvicorn app.main:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Docker Services

Redis, PostgreSQL, and Qdrant are included for local development:

```bash
docker compose up -d
```

## API Endpoints

### Health

```http
GET /health
```

### Document Ingestion

```http
POST /api/v1/documents/ingest
Content-Type: multipart/form-data
```

Form fields:

- `file`: `.pdf` or `.txt`
- `chunking_strategy`: `recursive` or `semantic`

### Conversational RAG

```http
POST /api/v1/chat/query
Content-Type: application/json
```

```json
{
  "session_id": "candidate-session-1",
  "message": "What does the uploaded document say about Python experience?"
}
```

### Interview Booking

Booking is handled through the same chat endpoint. Example:

```json
{
  "session_id": "candidate-session-1",
  "message": "Book my interview. My name is Aashik, email aashik@example.com, date 2026-07-01, time 10:30."
}
```

The LLM extracts booking details and stores them in the database when all required fields are present.

## Tests

```bash
pytest
```

