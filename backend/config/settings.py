"""
Application configuration - no hardcoded values.
All settings loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings from environment."""
    
    # Neo4j (NEO4J_USERNAME accepted as alias for neo4j_user)
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_username: str = ""  # Aura may use instance ID as username
    neo4j_password: str = ""
    
    # LLM
    llm_provider: str = "ollama"  # ollama | gemini | openai
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"
    ollama_timeout: float = 180.0  # seconds; graph extraction needs longer than 60s
    ollama_vision_model: str = ""  # Optional: llama3.2-vision for image description
    gemini_api_key: str = ""
    openai_api_key: str = ""
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_url: str = "http://localhost:3001"
    
    # Storage
    upload_dir: str = "./uploads"
    faiss_index_dir: str = "./faiss_index"
    log_level: str = "INFO"
    
    # Chunking
    chunk_min_tokens: int = 150
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 50
    chunk_similarity_threshold: float = 0.5  # Split when consecutive sentence similarity < this

    # Extraction (fast path: OCR only, no vision)
    enable_translation: bool = True  # Translate non-English content to English (skips if mostly ASCII)
    skip_table_llm: bool = True  # Skip LLM for table context
    extract_charts: bool = True  # Chart extraction via OCR only (fast)

    # Content deduplication (PDF text + OCR + Vision overlap)
    content_dedup_similarity_threshold: float = 0.85  # cosine sim >= this = same content, merge

    # Chunking quality
    fast_chunking: bool = False  # False = full structural + semantic + size/overlap
    # Graph-aware chunking: coherence (0–1 MiniLM-based); below = drop or split further
    chunk_coherence_min: float = 0.22
    chunk_entity_split_min_tokens: int = 80  # only refine splits when chunk is large enough
    chunk_max_entities_before_split: int = 4  # unrelated-entity pressure → try split

    # Graph build: hybrid NER + rules; LLM only when confidence below threshold
    graph_extract_concurrency: int = 5
    graph_llm_cache_max_entries: int = 4096
    graph_llm_fallback_threshold: float = 0.38  # below → optional LLM augment
    graph_use_spacy_ner: bool = False  # True: pip install spacy + en_core_web_sm
    graph_enable_communities: bool = True

    # Query: latency-optimized defaults
    query_retrieval_k: int = 8
    query_context_max_chunks: int = 3  # Top 3 for <3-5s latency
    query_context_max_chars_per_chunk: int = 400
    query_llm_timeout_sec: float = 15.0

    @property
    def neo4j_user_resolved(self) -> str:
        """Use neo4j_username if set (Aura), else neo4j_user."""
        return self.neo4j_username or self.neo4j_user

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
