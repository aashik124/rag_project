from enum import StrEnum

from pydantic import BaseModel


class ChunkingStrategy(StrEnum):
    recursive = "recursive"
    semantic = "semantic"


class DocumentIngestResponse(BaseModel):
    document_id: int
    filename: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    vector_store: str
