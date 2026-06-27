from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "Palm Mind RAG Backend"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "sqlite+aiosqlite:///./palm_mind.db"
    redis_url: str = "redis://localhost:6379/0"

    # Local embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Chat model used for RAG answers and booking extraction.
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"

    vector_store: Literal["qdrant", "pinecone"] = "qdrant"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    pinecone_api_key: str | None = None
    pinecone_index_name: str = "palm-mind-documents"

    rag_top_k: int = 5
    chat_memory_ttl_seconds: int = 86400


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()
