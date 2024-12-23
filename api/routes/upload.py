from fastapi import APIRouter, UploadFile, File
from core.document_processor import DocumentProcessor
from core.embeddings import DocumentEmbeddings
from core.vector_store import VectorStore
from models.schemas import UploadResponse
from app.config import settings
import uuid

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    file_content = await file.read()
    
    # Process document
    processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    # Extract and split text
    text = processor.extract_text(file_content, file.content_type)
    chunks = processor.split_text(text)
    
    # Create embeddings
    embedder = DocumentEmbeddings()
    embeddings = embedder.embed_documents(chunks)
    
    # Store in vector database
    vector_store = VectorStore(
        dimension=384,  # Dimension of all-MiniLM-L6-v2 embeddings
        index_path=settings.VECTOR_STORE_PATH
    )
    vector_store.create_index(embeddings, chunks)
    
    file_id = str(uuid.uuid4())
    return UploadResponse(
        file_id=file_id,
        status="success",
        message="File processed successfully"
    )