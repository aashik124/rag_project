from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingService:
    def __init__(self) -> None:
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded successfully!")

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()
