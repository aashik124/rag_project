from functools import lru_cache

from app.rag.pipeline import ConversationalRAGService
from app.rag.vector_store import VectorStore, get_vector_store
from app.services.booking import BookingService
from app.services.embeddings import EmbeddingService
from app.services.ingestion import DocumentIngestionService
from app.services.memory import RedisChatMemory


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_store() -> VectorStore:
    return get_vector_store()


@lru_cache
def get_memory() -> RedisChatMemory:
    return RedisChatMemory()


@lru_cache
def get_booking_service() -> BookingService:
    return BookingService()


def get_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService(get_embedding_service(), get_store())


def get_rag_service() -> ConversationalRAGService:
    return ConversationalRAGService(
        embeddings=get_embedding_service(),
        vector_store=get_store(),
        memory=get_memory(),
        booking_service=get_booking_service(),
    )
