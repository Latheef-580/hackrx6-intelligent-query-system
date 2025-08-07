import os
from typing import Optional

class Settings:
    # API Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    
    # Model Configuration
    USE_OPENAI_EMBEDDINGS: bool = bool(os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() == "true")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
    
    # LLM Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    MAX_RESPONSE_TOKENS: int = int(os.getenv("MAX_RESPONSE_TOKENS", "500"))
    
    # Document Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "300"))  # words per chunk
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))  # words overlap
    
    # Vector Search Configuration
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "IndexFlatIP")
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.3"))
    
    # Server Configuration
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # Authentication
    REQUIRE_AUTH: bool = bool(os.getenv("REQUIRE_AUTH", "true").lower() == "true")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "120"))  # seconds
    
    # Memory Management
    MAX_VECTOR_STORES: int = int(os.getenv("MAX_VECTOR_STORES", "10"))
    MAX_DOCUMENT_SIZE_MB: int = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50"))
    
    def __init__(self):
        # Validate critical settings
        if self.USE_OPENAI_EMBEDDINGS and not self.OPENAI_API_KEY:
            print("Warning: USE_OPENAI_EMBEDDINGS is True but OPENAI_API_KEY is not set. Falling back to local embeddings.")
            self.USE_OPENAI_EMBEDDINGS = False
        
        # Auto-adjust embedding dimension for OpenAI
        if self.USE_OPENAI_EMBEDDINGS and self.OPENAI_API_KEY:
            self.EMBEDDING_DIM = 1536  # OpenAI ada-002 dimension

# Global settings instance
settings = Settings()

# Environment-specific configurations
class DevelopmentConfig(Settings):
    LOG_LEVEL = "DEBUG"
    REQUIRE_AUTH = False

class ProductionConfig(Settings):
    LOG_LEVEL = "INFO"
    REQUIRE_AUTH = True
    WORKERS = 2

class TestingConfig(Settings):
    LOG_LEVEL = "WARNING"
    REQUIRE_AUTH = False
    USE_OPENAI_EMBEDDINGS = False

# Configuration factory
def get_config():
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Override settings if needed
if os.getenv("ENVIRONMENT"):
    settings = get_config()