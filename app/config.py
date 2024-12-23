from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MISTRAL_API_KEY: str
    VECTOR_STORE_PATH: str = "vector_store/faiss_index"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MODEL_NAME: str = "mistral-large-latest"
    
    class Config:
        env_file = ".env"

settings = Settings()