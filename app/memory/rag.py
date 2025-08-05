"""Full RAG logic pipeline."""
from .vector_store import VectorStore
from .embedder import Embedder

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = Embedder()
    
    def add_knowledge(self, text: str, metadata: dict = None):
        """Add knowledge to RAG system."""
        embedding = self.embedder.embed_text(text)
        doc_id = f"doc_{len(self.vector_store.vectors)}"
        self.vector_store.add_document(doc_id, embedding, metadata or {})
    
    def retrieve_context(self, query: str, k: int = 3) -> list:
        """Retrieve relevant context for query."""
        query_embedding = self.embedder.embed_text(query)
        return self.vector_store.search(query_embedding, k)