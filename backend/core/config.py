"""
Configuration module for Maritime Q&A Assistant
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
from pathlib import Path

# Find .env file - look in current dir, parent dir, and project root
def find_env_file():
    """Find .env file in current, parent, or project root directory"""
    current = Path(__file__).resolve().parent
    
    # Try current directory (core/)
    if (current / ".env").exists():
        return str(current / ".env")
    
    # Try parent directory (backend/)
    if (current.parent / ".env").exists():
        return str(current.parent / ".env")
    
    # Try grandparent directory (project root)
    if (current.parent.parent / ".env").exists():
        return str(current.parent.parent / ".env")
    
    # Default to parent
    return str(current.parent / ".env")


ENV_FILE = find_env_file()


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_title: str = "Maritime Documentation Q&A Assistant"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str  # REQUIRED: Set in .env file
    neo4j_database: str = "neo4j"
    
    # LLM Provider Configuration
    llm_provider: str = "groq"  # "openai", "groq", or "cerebras"
    
    # OpenAI Configuration
    openai_api_key: str  # REQUIRED: Set in .env file
    llm_model: str = "openai/gpt-oss-120b"  # Default model (change based on provider)
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Groq Configuration
    groq_api_key: Optional[str] = None  # REQUIRED if llm_provider=groq
    
    # Cerebras Configuration
    cerebras_api_key: Optional[str] = None  # REQUIRED if llm_provider=cerebras
    cerebras_base_url: str = "https://api.cerebras.ai/v1"  # Cerebras API endpoint
    
    # Storage Configuration
    storage_type: str = "local"  # "local" or "s3"
    local_storage_path: str = "../../data"
    s3_bucket_name: Optional[str] = None
    s3_prefix: Optional[str] = None  # S3 folder prefix (e.g., 'data' if files are in data/schemas/...)
    aws_region: Optional[str] = "eu-central-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # Processing Configuration
    chunk_size: int = 50  # Pages to process at once
    max_file_size: int = 200 * 1024 * 1024  # 200MB
    supported_languages: list = ["en", ]
    
    # Vision API Cost Optimization
    vision_detail_tables: str = "high"   # "low" | "high" | "auto" - High needed for small text in tables
    vision_detail_schemas: str = "auto"  # "low" | "high" | "auto" - Auto adapts based on image size
    
    # Vector Database / Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_use_https: bool = True  # True for Qdrant Cloud, False for local

    text_chunks_collection: str = "text_chunks"
    schemas_collection: str = "schemas"
    tables_text_collection: str = "tables"

    # Embeddings / vector dim
    vector_dimension: int = 1536  # dim for text embeddings
    vector_index_type: str = "cosine"
    
    # Cache Configuration
    redis_url: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour
    
    # Authentication Configuration
    admin_username: str = "admin"
    admin_password: Optional[str] = None  # Required on first run if no credentials.json
    auth_secret_key: str 
    # Performance Configuration
    max_workers: int = 4
    batch_size: int = 10
    query_timeout: int = 30
    
    # Alias property for compatibility with workflow.py
    @property
    def embedding_model(self) -> str:
        """Alias for openai_embedding_model used in workflow.py"""
        return self.openai_embedding_model
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,  # Dynamically found .env file
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )


settings = Settings()


# Debug print to verify config is loaded
if __name__ == "__main__":
    print("=" * 80)
    print("CONFIGURATION LOADED")
    print("=" * 80)
    print(f"ENV file: {ENV_FILE}")
    print(f"ENV file exists: {Path(ENV_FILE).exists()}")
    print()
    print(f"Neo4j URI: {settings.neo4j_uri}")
    print(f"Neo4j User: {settings.neo4j_user}")
    print(f"Neo4j Database: {settings.neo4j_database}")
    print()
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"OpenAI API Key: {'*' * 20}{settings.openai_api_key[-8:] if len(settings.openai_api_key) > 8 else '***'}")
    if settings.groq_api_key:
        print(f"Groq API Key: {'*' * 20}{settings.groq_api_key[-8:] if len(settings.groq_api_key) > 8 else '***'}")
    if settings.cerebras_api_key:
        print(f"Cerebras API Key: {'*' * 20}{settings.cerebras_api_key[-8:] if len(settings.cerebras_api_key) > 8 else '***'}")
        print(f"Cerebras Base URL: {settings.cerebras_base_url}")
    print(f"LLM Model: {settings.llm_model}")
    print(f"Embedding Model: {settings.openai_embedding_model}")
    print()
    print(f"Qdrant Host: {settings.qdrant_host}")
    print(f"Qdrant Port: {settings.qdrant_port}")
    print()
    print(f"Storage Type: {settings.storage_type}")
    print(f"Storage Path: {settings.local_storage_path}")
    print("=" * 80)