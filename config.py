"""Configuration management for the RAG system."""
import os
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class DetailLevel(str, Enum):
    """Summary detail levels."""
    EXECUTIVE = "ejecutivo"
    NORMAL = "normal"
    DETAILED = "detallado"


class StorageType(str, Enum):
    """Storage backend types."""
    LOCAL = "local"
    CLOUD = "cloud"


class Settings(BaseSettings):
    """Application settings."""

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)

    # Google Cloud
    gcp_project_id: str = Field(default="")
    gcp_location: str = Field(default="us-central1")

    # Vertex AI
    vertex_ai_model: str = Field(default="gemini-2.5-pro")
    embedding_model: str = Field(default="text-embedding-004")
    vector_search_index_id: str = Field(default="")
    vector_search_endpoint_id: str = Field(default="")

    # Storage
    storage_type: StorageType = Field(default=StorageType.LOCAL)
    local_pdf_path: str = Field(default="./data/pdfs")
    local_output_path: str = Field(default="./data/outputs")
    local_temp_path: str = Field(default="./data/temp")
    gcs_bucket_name: str = Field(default="")
    gcs_output_bucket: str = Field(default="")

    # Processing
    max_pdf_pages: int = Field(default=2000)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    max_concurrent_requests: int = Field(default=5)

    # Application
    port: int = Field(default=8501)

    # Redis (optional)
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)

    class Config:
        """Pydantic config."""
        env_file = ".env.development"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    env_file = f".env.{env}"

    if os.path.exists(env_file):
        return Settings(_env_file=env_file)
    return Settings()


# Global settings instance
settings = get_settings()
