from fastapi import APIRouter
from core.embeddings import DocumentEmbeddings
from core.vector_store import VectorStore
from core.chat_model import ChatModel
from models.schemas import ChatRequest, ChatResponse
from app.config import settings
import uuid

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Load vector store
    vector_store = VectorStore(
        dimension=384,
        index_path=settings.VECTOR_STORE_PATH
    )
    vector_store.load_index()
    
    # Get embeddings for question
    embedder = DocumentEmbeddings()
    question_embedding = embedder.embed_query(request.question)
    
    # Search similar contexts
    relevant_chunks = vector_store.search(question_embedding)
    
    # Get response from Mistral
    chat_model = ChatModel()
    answer = chat_model.get_response(relevant_chunks, request.question)
    
    return ChatResponse(
        answer=answer,
        conversation_id=request.conversation_id or str(uuid.uuid4())
    )