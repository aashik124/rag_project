from io import BytesIO

from fastapi import UploadFile
from pypdf import PdfReader


class UnsupportedFileTypeError(ValueError):
    pass


async def extract_text_from_upload(file: UploadFile) -> str:
    content = await file.read()
    filename = file.filename or ""

    if filename.lower().endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(page.strip() for page in pages if page.strip())

    raise UnsupportedFileTypeError("Only .pdf and .txt files are supported.")
