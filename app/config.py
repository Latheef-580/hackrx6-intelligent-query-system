import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Team token for authentication
    TEAM_TOKEN: str = "b101776f72f459eca15614eb73a6f17efe85d475b21adb16c794068573018565"
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-3.5-turbo"
    MAX_RESPONSE_TOKENS: int = 500
    
    # Embedding Configuration
    USE_OPENAI_EMBEDDINGS: bool = False
    EMBEDDING_MODEL: str = "all-MiniLM-L3-v2"
    EMBEDDING_DIM: int = 384
    
    # Memory optimization
    USE_FALLBACK_MODE: bool = True  # Use simple text matching instead of embeddings
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CHUNKS: int = 100  # Limit chunks for memory
    
    # Environment
    ENVIRONMENT: str = "production"
    
    class Config:
        env_file = ".env"

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