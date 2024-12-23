from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str

class UploadResponse(BaseModel):
    file_id: str
    status: str
    message: str