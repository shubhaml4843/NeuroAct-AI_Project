"""Pre-load RAG memory with initial knowledge."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.memory.rag import RAGPipeline

def seed_knowledge():
    print("Seeding vector store with initial knowledge...")
    
    rag = RAGPipeline()
    
    # Add sample knowledge
    knowledge_base = [
        "Python is a high-level programming language",
        "Machine learning involves training models on data",
        "FastAPI is a modern web framework for Python",
        "Vector databases store high-dimensional embeddings"
    ]
    
    for i, knowledge in enumerate(knowledge_base):
        rag.add_knowledge(knowledge, {"source": f"seed_{i}", "category": "general"})
    
    print(f"Added {len(knowledge_base)} knowledge entries to vector store")

if __name__ == "__main__":
    seed_knowledge()