import json
from dataclasses import asdict, dataclass

from redis.asyncio import Redis

from app.core.config import settings


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class RedisChatMemory:
    def __init__(self) -> None:
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    async def get_messages(self, session_id: str, limit: int = 10) -> list[ChatMessage]:
        values = await self.redis.lrange(self._key(session_id), -limit, -1)
        return [ChatMessage(**json.loads(value)) for value in values]

    async def append(self, session_id: str, message: ChatMessage) -> None:
        key = self._key(session_id)
        await self.redis.rpush(key, json.dumps(asdict(message)))
        await self.redis.expire(key, settings.chat_memory_ttl_seconds)
