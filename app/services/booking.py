import json
from datetime import date, time

import google.generativeai as genai
from pydantic import BaseModel, EmailStr, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.booking import InterviewBooking


class BookingDetails(BaseModel):
    wants_booking: bool = False
    name: str | None = None
    email: EmailStr | None = None
    interview_date: date | None = None
    interview_time: time | None = None

    @property
    def is_complete(self) -> bool:
        return bool(self.name and self.email and self.interview_date and self.interview_time)


class BookingService:
    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_chat_model,
            generation_config={"response_mime_type": "application/json"},
        )

    async def extract_booking_details(self, message: str, history: str) -> BookingDetails:
        system = (
            "Extract interview booking information from the conversation. "
            "Return strict JSON with keys: wants_booking, name, email, interview_date, interview_time. "
            "Use ISO date YYYY-MM-DD and 24-hour HH:MM:SS time. Use null for missing values."
        )
        prompt = f"{system}\n\nConversation history:\n{history}\n\nLatest message:\n{message}"
        response = await self.model.generate_content_async(prompt)
        content = response.text or "{}"
        try:
            return BookingDetails.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError):
            return BookingDetails()

    async def store_booking(self, session: AsyncSession, session_id: str, details: BookingDetails) -> InterviewBooking:
        if not details.is_complete:
            raise ValueError("Booking details are incomplete.")
        booking = InterviewBooking(
            session_id=session_id,
            name=details.name or "",
            email=str(details.email),
            interview_date=details.interview_date,
            interview_time=details.interview_time,
        )
        session.add(booking)
        await session.commit()
        await session.refresh(booking)
        return booking
