"""SentenceTransformer / OpenAI API embeddings."""

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
    
    def embed_text(self, text: str) -> list:
        """Generate embedding for text."""
        return [0.1] * 384  # Mock embedding
    
    def embed_batch(self, texts: list) -> list:
        """Generate embeddings for batch of texts."""
        return [[0.1] * 384 for _ in texts]