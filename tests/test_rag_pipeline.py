"""Tests for RAG pipeline."""
import unittest
from app.memory.rag import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    def test_add_knowledge(self):
        rag = RAGPipeline()
        rag.add_knowledge("Test knowledge", {"source": "test"})
        self.assertEqual(len(rag.vector_store.vectors), 1)
    
    def test_retrieve_context(self):
        rag = RAGPipeline()
        rag.add_knowledge("Test knowledge", {"source": "test"})
        results = rag.retrieve_context("test query")
        self.assertIsInstance(results, list)