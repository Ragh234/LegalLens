from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration for the LegalLens RAG pipeline."""

    app_name: str = "LegalLens"
    data_dir: Path = Path("data")
    contracts_dir: Path = Path("data/contracts")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legallens_contract_chunks"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "gemini-1.5-flash"
    gemini_api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    log_level: str = "INFO"
    vector_size: int = 384
    upload_max_mb: int = Field(default=25, ge=1)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
