"""API keys, constants, and configuration."""
import os

# API Keys (use environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Model Configuration
DEFAULT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# System Configuration
MAX_AGENTS = 10
DEFAULT_TIMEOUT = 300
VECTOR_STORE_PATH = "./data/vector_store"

# Reinforcement Learning
LEARNING_RATE = 0.001
REWARD_THRESHOLD = 0.8