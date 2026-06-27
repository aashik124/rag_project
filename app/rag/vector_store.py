from dataclasses import dataclass
from typing import Protocol

from pinecone import Pinecone, ServerlessSpec
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.core.config import settings


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    text: str
    score: float | None
    metadata: dict[str, str | int]


class VectorStore(Protocol):
    async def ensure_collection(self) -> None:
        ...

    async def upsert(self, records: list[VectorRecord]) -> None:
        ...

    async def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        ...


class QdrantVectorStore:
    def __init__(self) -> None:
        self.client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.collection = settings.qdrant_collection

    async def ensure_collection(self) -> None:
        collections = await self.client.get_collections()
        names = {collection.name for collection in collections.collections}
        if self.collection not in names:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=settings.embedding_dimension, distance=Distance.COSINE),
            )

    async def upsert(self, records: list[VectorRecord]) -> None:
        await self.ensure_collection()
        points = [
            PointStruct(
                id=record.id,
                vector=record.vector,
                payload={"text": record.text, **record.metadata},
            )
            for record in records
        ]
        await self.client.upsert(collection_name=self.collection, points=points)

    async def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        await self.ensure_collection()
        results = await self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=Filter(must_not=[FieldCondition(key="deleted", match=MatchValue(value=True))]),
        )
        return [
            VectorSearchResult(
                id=str(result.id),
                text=str((result.payload or {}).get("text", "")),
                score=result.score,
                metadata={k: v for k, v in (result.payload or {}).items() if k != "text"},
            )
            for result in results
        ]


class PineconeVectorStore:
    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required when VECTOR_STORE=pinecone")
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name

    async def ensure_collection(self) -> None:
        existing = {index["name"] for index in self.pc.list_indexes()}
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    async def upsert(self, records: list[VectorRecord]) -> None:
        await self.ensure_collection()
        index = self.pc.Index(self.index_name)
        vectors = [
            {
                "id": record.id,
                "values": record.vector,
                "metadata": {"text": record.text, **record.metadata},
            }
            for record in records
        ]
        index.upsert(vectors=vectors)

    async def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        await self.ensure_collection()
        index = self.pc.Index(self.index_name)
        response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return [
            VectorSearchResult(
                id=match["id"],
                text=str((match.get("metadata") or {}).get("text", "")),
                score=match.get("score"),
                metadata={k: v for k, v in (match.get("metadata") or {}).items() if k != "text"},
            )
            for match in response.get("matches", [])
        ]


def get_vector_store() -> VectorStore:
    if settings.vector_store == "pinecone":
        return PineconeVectorStore()
    return QdrantVectorStore()
