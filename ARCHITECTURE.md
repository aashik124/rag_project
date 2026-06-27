# Architecture

## Request Flow

### Document Ingestion

1. `POST /api/v1/documents/ingest`
2. `text_extractor.py` extracts text from `.pdf` or `.txt`
3. `chunking.py` applies `recursive` or `semantic` strategy
4. `embeddings.py` generates local sentence-transformers embeddings
5. `vector_store.py` stores vectors in Qdrant or Pinecone
6. SQLAlchemy stores document and chunk metadata

### Conversational RAG

1. `POST /api/v1/chat/query`
2. Redis loads recent chat turns by `session_id`
3. The latest message and history are embedded
4. Qdrant or Pinecone returns relevant chunks
5. `pipeline.py` builds a custom prompt and calls the LLM
6. User and assistant messages are written back to Redis

### Booking

1. The chat API sends the latest message and history to `BookingService`
2. The LLM returns structured JSON booking fields
3. Complete booking details are stored in `interview_bookings`
4. Missing fields are requested through the normal chat response

## Important Constraint

The RAG pipeline is custom code in `app/rag/pipeline.py`. It does not use LangChain `RetrievalQAChain`.
