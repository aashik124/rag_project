from datetime import date, time

from pydantic import BaseModel, EmailStr, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1)


class SourceDocument(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    score: float | None = None
    text: str


class BookingResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    interview_date: date
    interview_time: time


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceDocument] = []
    booking: BookingResponse | None = None
