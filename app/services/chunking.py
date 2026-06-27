import re
from dataclasses import dataclass

from app.schemas.document import ChunkingStrategy


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str


class Chunker:
    def chunk(self, text: str) -> list[TextChunk]:
        raise NotImplementedError


class RecursiveChunker(Chunker):
    def __init__(self, chunk_size: int = 1_000, overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[TextChunk]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            candidate = normalized[start:end]
            split_at = max(candidate.rfind(". "), candidate.rfind("\n"), candidate.rfind(" "))
            if split_at > self.chunk_size * 0.5 and end < len(normalized):
                end = start + split_at + 1
                candidate = normalized[start:end]
            chunks.append(candidate.strip())
            if end >= len(normalized):
                break
            start = max(end - self.overlap, 0)

        return [TextChunk(index=i, text=chunk) for i, chunk in enumerate(chunks) if chunk]


class SemanticChunker(Chunker):
    def __init__(self, target_size: int = 900) -> None:
        self.target_size = target_size

    def chunk(self, text: str) -> list[TextChunk]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units = paragraphs or [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        grouped: list[str] = []
        current = ""
        for unit in units:
            if len(current) + len(unit) + 2 <= self.target_size:
                current = f"{current}\n\n{unit}".strip()
            else:
                if current:
                    grouped.append(current)
                current = unit
        if current:
            grouped.append(current)

        return [TextChunk(index=i, text=chunk) for i, chunk in enumerate(grouped) if chunk]


def get_chunker(strategy: ChunkingStrategy) -> Chunker:
    if strategy is ChunkingStrategy.semantic:
        return SemanticChunker()
    return RecursiveChunker()
