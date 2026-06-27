from app.schemas.document import ChunkingStrategy
from app.services.chunking import get_chunker


def test_recursive_chunker_splits_long_text() -> None:
    text = " ".join(["FastAPI makes backend services clean."] * 120)
    chunks = get_chunker(ChunkingStrategy.recursive).chunk(text)

    assert len(chunks) > 1
    assert chunks[0].index == 0
    assert all(chunk.text for chunk in chunks)


def test_semantic_chunker_keeps_paragraphs() -> None:
    text = "First paragraph about RAG.\n\nSecond paragraph about Redis memory."
    chunks = get_chunker(ChunkingStrategy.semantic).chunk(text)

    assert len(chunks) == 1
    assert "Redis memory" in chunks[0].text
