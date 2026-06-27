import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.rag.vector_store import VectorSearchResult, VectorStore
from app.schemas.chat import BookingResponse, ChatResponse, SourceDocument
from app.services.booking import BookingService
from app.services.embeddings import EmbeddingService
from app.services.memory import ChatMessage, RedisChatMemory


class ConversationalRAGService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        memory: RedisChatMemory,
        booking_service: BookingService,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.memory = memory
        self.booking_service = booking_service

        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(model_name=settings.gemini_chat_model)

    async def answer(self, session: AsyncSession, session_id: str, message: str) -> ChatResponse:
        history = await self.memory.get_messages(session_id)
        history_text = self._format_history(history)

        booking_details = await self.booking_service.extract_booking_details(message, history_text)
        if booking_details.wants_booking and booking_details.is_complete:
            booking = await self.booking_service.store_booking(session, session_id, booking_details)
            answer = (
                f"Your interview has been booked for {booking.interview_date.isoformat()} "
                f"at {booking.interview_time.strftime('%H:%M')}."
            )
            await self._remember(session_id, message, answer)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                booking=BookingResponse.model_validate(booking, from_attributes=True),
            )

        query_vector = await self.embeddings.embed_query(self._rewrite_query(message, history))
        search_results = await self.vector_store.search(query_vector, top_k=settings.rag_top_k)
        answer = await self._generate_answer(message, history_text, search_results, booking_details.wants_booking)

        await self._remember(session_id, message, answer)
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=[
                SourceDocument(
                    document_id=int(result.metadata.get("document_id", 0)),
                    filename=str(result.metadata.get("filename", "")),
                    chunk_index=int(result.metadata.get("chunk_index", 0)),
                    score=result.score,
                    text=result.text,
                )
                for result in search_results
            ],
        )

    def _rewrite_query(self, message: str, history: list[ChatMessage]) -> str:
        recent = " ".join(item.content for item in history[-4:])
        return f"{recent} {message}".strip()

    def _format_history(self, history: list[ChatMessage]) -> str:
        return "\n".join(f"{item.role}: {item.content}" for item in history)

    async def _remember(self, session_id: str, user_message: str, assistant_message: str) -> None:
        await self.memory.append(session_id, ChatMessage(role="user", content=user_message))
        await self.memory.append(session_id, ChatMessage(role="assistant", content=assistant_message))

    async def _generate_answer(
        self,
        message: str,
        history: str,
        contexts: list[VectorSearchResult],
        wants_booking: bool,
    ) -> str:
        context_text = "\n\n".join(
            f"[Source {i + 1}] {result.text}" for i, result in enumerate(contexts)
        )
        booking_instruction = (
            "If the user is trying to book an interview but details are missing, ask only for the missing "
            "name, email, date, or time."
            if wants_booking
            else ""
        )
        system_prompt = (
            "You are a helpful RAG assistant. Answer using the provided context when relevant. "
            "Do not invent facts. If the context is insufficient, say what is missing. "
            f"{booking_instruction}"
        )
        full_prompt = (
            f"{system_prompt}\n\n"
            f"History:\n{history}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{message}"
        )
        response = await self.model.generate_content_async(full_prompt)
        return response.text or "I could not generate an answer."
