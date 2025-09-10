"""Configuration management for NeuroAct AI with Ollama backend."""
import os
from pathlib import Path
from dataclasses import dataclass

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
VECTOR_DIR = DATA_DIR / "vectors"
LOG_DIR = ROOT_DIR / "logs"

@dataclass(frozen=True)
class Settings:
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    
    # Alternative models for different tasks
    CODE_MODEL: str = os.getenv("CODE_MODEL", "codellama:7b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    # Optional API Keys (for tools that need them)
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    
    # Vector Store Configuration
    VECTOR_DB: str = os.getenv("VECTOR_DB", "faiss")
    VECTOR_DIR: str = os.getenv("VECTOR_DIR", str(VECTOR_DIR))
    
    # System Configuration
    MAX_AGENTS: int = int(os.getenv("MAX_AGENTS", "10"))
    DEFAULT_TIMEOUT: int = int(os.getenv("DEFAULT_TIMEOUT", "300"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
    MAX_TOOL_STEPS: int = int(os.getenv("MAX_TOOL_STEPS", "8"))
    
    # Reinforcement Learning
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
    REWARD_THRESHOLD: float = float(os.getenv("REWARD_THRESHOLD", "0.8"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", str(LOG_DIR))
    
    # Feature Flags
    ENABLE_RL: bool = os.getenv("ENABLE_RL", "true").lower() == "true"
    ENABLE_RAG: bool = os.getenv("ENABLE_RAG", "true").lower() == "true"
    ENABLE_MCP: bool = os.getenv("ENABLE_MCP", "true").lower() == "true"
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8700"))
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"

# Global settings instance
settings = Settings()

def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    Path(settings.VECTOR_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize on import
ensure_dirs()

# Helper functions
def get_ollama_config() -> dict:
    """Get Ollama configuration."""
    return {
        "base_url": settings.OLLAMA_BASE_URL,
        "model": settings.OLLAMA_MODEL,
        "code_model": settings.CODE_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "timeout": settings.OLLAMA_TIMEOUT
    }

def model_name() -> str:
    """Return the current LLM name."""
    return settings.OLLAMA_MODEL

def embedding_model_name() -> str:
    """Return the current embedding model name."""
    return settings.EMBEDDING_MODEL

def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled."""
    feature_map = {
        "rl": settings.ENABLE_RL,
        "rag": settings.ENABLE_RAG,
        "mcp": settings.ENABLE_MCP
    }
    return feature_map.get(feature.lower(), False)