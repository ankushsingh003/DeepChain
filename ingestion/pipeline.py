"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Data Ingestion Pipeline Orchestrator
"""

import json
from typing import List
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from graph.extractor import TripletExtractor
from graph.neo4j_client import Neo4jClient
from graph.builder import GraphBuilder
from vector_store.weaviate_client import WeaviateClient
from vector_store.embedder import GeminiEmbedder

class IngestionPipeline:
    def __init__(self, data_path: str = "data/sample_docs"):
        self.loader = DocumentLoader(data_path)
        self.chunker = DocumentChunker()
        self.extractor = TripletExtractor()
        self.neo4j_client = Neo4jClient()
        self.weaviate_client = WeaviateClient()
        self.embedder = GeminiEmbedder()

    def run(self):
        """Runs the full ingestion pipeline: Load -> Chunk -> Extract -> Store."""
        print("\n[PIPELINE] Starting Information Extraction Pipeline...")
        
        # 1. Load
        documents = self.loader.load_documents()
        if not documents:
            print("[!] No documents found. Exiting.")
            return

        # 2. Chunk
        chunks = self.chunker.split_documents(documents)
        chunks_data = [
            {
                "text": c.page_content,
                "source": c.metadata.get("source", "unknown"),
                "chunk_id": f"chunk_{i}",
            }
            for i, c in enumerate(chunks)
        ]
        
        # 3. Extract Triplets (New Logic)
        print(f"[*] Extracting triplets from {len(chunks_data)} chunks...")
        triplets = self.extractor.extract(chunks_data)
        
        # 4. Store in Neo4j
        print(f"[*] Building Knowledge Graph in Neo4j...")
        self.neo4j_client.initialize_schema()
        # We need to adapt triplets to the KnowledgeGraph schema if builder expects it
        # Or update builder. For now, let's convert triplets to the expected format
        from ingestion.extractor import KnowledgeGraph, Entity, Relationship
        
        entities_map = {}
        relationships = []
        for t in triplets:
            subj = t["subject"]
            obj = t["object"]
            pred = t["predicate"]
            
            if subj not in entities_map:
                entities_map[subj] = Entity(name=subj, type="Entity", description="Extracted entity")
            if obj not in entities_map:
                entities_map[obj] = Entity(name=obj, type="Entity", description="Extracted entity")
                
            relationships.append(Relationship(
                source=subj, 
                target=obj, 
                type=pred, 
                description=f"Extracted from {t.get('source_chunk_id')}"
            ))
            
        kg = KnowledgeGraph(entities=list(entities_map.values()), relationships=relationships)
        builder = GraphBuilder(self.neo4j_client)
        builder.build_graph(kg)
        
        # 5. Store in Weaviate
        print(f"[*] Storing {len(chunks_data)} chunks in Weaviate...")
        # Note: Weaviate expects 'content' instead of 'text' based on previous schema
        # Let's align them
        weaviate_data = []
        texts = []
        for c in chunks_data:
            weaviate_data.append({
                "content": c["text"],
                "source": c["source"],
                "chunk_id": int(c["chunk_id"].split("_")[1])
            })
            texts.append(c["text"])
            
        vectors = self.embedder.embed_documents(texts)
        self.weaviate_client.upsert_chunks(weaviate_data, vectors)
        
        print(f"\n[PIPELINE] Completed successfully.")

if __name__ == "__main__":
    # Ensure sample data exists
    import os
    os.makedirs("data/sample_docs", exist_ok=True)
    pipeline = IngestionPipeline()
    pipeline.run()
