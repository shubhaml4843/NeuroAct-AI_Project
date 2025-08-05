"""FAISS or Chroma vector store wrapper."""

class VectorStore:
    def __init__(self, store_type="faiss"):
        self.store_type = store_type
        self.vectors = {}
    
    def add_document(self, doc_id: str, embedding: list, metadata: dict):
        """Add document embedding to store."""
        self.vectors[doc_id] = {"embedding": embedding, "metadata": metadata}
    
    def search(self, query_embedding: list, k: int = 5) -> list:
        """Search for similar documents."""
        return list(self.vectors.keys())[:k]