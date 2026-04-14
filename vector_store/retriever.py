"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Vector Retriever Interface
"""

from typing import List, Dict, Any
from vector_store.weaviate_client import WeaviateClient
from vector_store.embedder import GeminiEmbedder

class VectorRetriever:
    def __init__(self, client: WeaviateClient, embedder: GeminiEmbedder):
        self.client = client
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant document chunks for a query."""
        print(f"[*] Vector Search: Finding top {top_k} results for '{query}'...")
        query_vector = self.embedder.embed_query(query)
        results = self.client.search(query_vector, limit=top_k)
        
        return [
            {
                "content": obj.properties["content"],
                "source": obj.properties["source"],
                "chunk_id": obj.properties["chunk_id"]
            }
            for obj in results
        ]

if __name__ == "__main__":
    # Test Retriever
    w_client = WeaviateClient()
    embedder = GeminiEmbedder()
    retriever = VectorRetriever(w_client, embedder)
    
    # This assumes data exists in Weaviate
    # hits = retriever.retrieve("What is Novatech?")
    # for hit in hits:
    #     print(f" - {hit['content'][:100]}...")
    w_client.close()
