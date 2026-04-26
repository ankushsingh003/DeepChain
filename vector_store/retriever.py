# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: Vector Retriever Interface
# """

# from typing import List, Dict, Any
# from vector_store.weaviate_client import WeaviateClient
# from vector_store.embedder import GeminiEmbedder

# class VectorRetriever:
#     def __init__(self, client: WeaviateClient, embedder: GeminiEmbedder):
#         self.client = client
#         self.embedder = embedder

#     def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Retrieves top_k relevant document chunks for a query."""
#         print(f"[*] Vector Search: Finding top {top_k} results for '{query}'...")
#         query_vector = self.embedder.embed_query(query)
#         results = self.client.search(query_vector, limit=top_k)
        
#         return [
#             {
#                 "content": obj.properties["content"],
#                 "source": obj.properties["source"],
#                 "chunk_id": obj.properties["chunk_id"]
#             }
#             for obj in results
#         ]

# if __name__ == "__main__":
#     # Test Retriever
#     w_client = WeaviateClient()
#     embedder = GeminiEmbedder()
#     retriever = VectorRetriever(w_client, embedder)
    
#     # This assumes data exists in Weaviate
#     # hits = retriever.retrieve("What is Novatech?")
#     # for hit in hits:
#     #     print(f" - {hit['content'][:100]}...")
#     w_client.close()







"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Vector Retriever Interface — Production Grade

Fixes & Additions over v1:
  - Distance threshold → rejects semantically irrelevant results
  - Metadata filter support (doc_type, language, source, custom)
  - Returns distance score alongside each result for transparency
  - Multi-tenant retrieval support
  - Async retrieval path for concurrent multi-query workloads
  - Deduplication of results by chunk_id (for hybrid/multi-query scenarios)
  - Full logging with timing metrics
  - Type hints throughout
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from weaviate.classes.query import Filter

from vector_store.weaviate_client import WeaviateClient, filter_by_doc_type, filter_by_language, filter_by_source
from vector_store.embedder import GeminiEmbedder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class RetrievedChunk:
    """
    Structured result object from a vector retrieval call.

    Attributes:
        content:      Raw chunk text.
        source:       Origin document identifier.
        chunk_id:     Position index within the source document.
        score:        Similarity score (1 - cosine_distance). Higher = more similar.
        distance:     Raw cosine distance from Weaviate.
        doc_type:     Document type (e.g. "pdf").
        page_number:  Page number within the source (if available).
        section:      Section heading (if available).
        language:     Language code (if available).
        token_count:  Approximate token count (if available).
        created_at:   Ingestion timestamp (if available).
    """

    def __init__(self, properties: Dict[str, Any], distance: float):
        self.content     = properties.get("content", "")
        self.source      = properties.get("source", "")
        self.chunk_id    = properties.get("chunk_id", -1)
        self.doc_type    = properties.get("doc_type", "")
        self.page_number = properties.get("page_number", None)
        self.section     = properties.get("section", "")
        self.language    = properties.get("language", "")
        self.token_count = properties.get("token_count", None)
        self.created_at  = properties.get("created_at", "")
        self.distance    = distance
        self.score       = round(1.0 - distance, 4)  # cosine similarity proxy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content":     self.content,
            "source":      self.source,
            "chunk_id":    self.chunk_id,
            "doc_type":    self.doc_type,
            "page_number": self.page_number,
            "section":     self.section,
            "language":    self.language,
            "token_count": self.token_count,
            "created_at":  self.created_at,
            "score":       self.score,
            "distance":    self.distance,
        }

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"RetrievedChunk(score={self.score}, source='{self.source}', "
            f"chunk_id={self.chunk_id}, content='{preview}...')"
        )


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------

class VectorRetriever:
    """
    Production-grade vector retriever for DeepChain-Hybrid-RAG.

    Features:
        - Distance threshold to suppress low-relevance results
        - Metadata pre-filtering (doc_type, language, source, or custom Filter)
        - Score attached to every result for downstream reranking
        - Async multi-query retrieval
        - Result deduplication (useful when combining multiple retrieval strategies)
    """

    def __init__(
        self,
        client: WeaviateClient,
        embedder: GeminiEmbedder,
        default_top_k: int = 5,
        default_distance_threshold: float = 0.30,
    ):
        self.client = client
        self.embedder = embedder
        self.default_top_k = default_top_k
        self.default_distance_threshold = default_distance_threshold

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        doc_type: Optional[str] = None,
        language: Optional[str] = None,
        source: Optional[str] = None,
        custom_filter: Optional[Filter] = None,
        tenant: Optional[str] = None,
        deduplicate: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most semantically relevant chunks for a query.

        Args:
            query:              Natural language query string.
            top_k:              Max results (defaults to self.default_top_k).
            distance_threshold: Cosine distance ceiling (defaults to self.default_distance_threshold).
            doc_type:           Restrict search to a specific document type.
            language:           Restrict search to a specific language.
            source:             Restrict search to a specific source document.
            custom_filter:      Any arbitrary Weaviate Filter for advanced filtering.
            tenant:             Tenant name for multi-tenant deployments.
            deduplicate:        Remove results with duplicate (source, chunk_id) pairs.

        Returns:
            List of RetrievedChunk objects, sorted by descending similarity score.
        """
        top_k = top_k or self.default_top_k
        distance_threshold = distance_threshold if distance_threshold is not None else self.default_distance_threshold

        t0 = time.perf_counter()
        logger.info(f"[VectorRetriever] Query='{query[:80]}' top_k={top_k} dist≤{distance_threshold}")

        # Embed the query
        query_vector = self.embedder.embed_query(query)

        # Build metadata filter (combine if multiple provided)
        active_filter = self._build_filter(doc_type, language, source, custom_filter)

        # Search Weaviate
        raw_results = self.client.search(
            vector=query_vector,
            limit=top_k,
            distance_threshold=distance_threshold,
            filters=active_filter,
            tenant=tenant,
        )

        # Map to RetrievedChunk objects
        chunks = [
            RetrievedChunk(
                properties=obj.properties,
                distance=obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 1.0,
            )
            for obj in raw_results
        ]

        # Deduplicate by (source, chunk_id)
        if deduplicate:
            chunks = self._deduplicate(chunks)

        # Sort by score descending (nearest first)
        chunks.sort(key=lambda c: c.score, reverse=True)

        elapsed = time.perf_counter() - t0
        logger.info(f"[VectorRetriever] Returned {len(chunks)} chunks in {elapsed:.3f}s")

        return chunks

    # ------------------------------------------------------------------
    # Async multi-query retrieval
    # ------------------------------------------------------------------

    async def retrieve_async(
        self,
        query: str,
        **kwargs,
    ) -> List[RetrievedChunk]:
        """Async wrapper around retrieve() — runs blocking call in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.retrieve(query, **kwargs))

    async def retrieve_multi(
        self,
        queries: List[str],
        **kwargs,
    ) -> List[List[RetrievedChunk]]:
        """
        Retrieve results for multiple queries concurrently.

        Args:
            queries: List of query strings.
            **kwargs: Passed to each retrieve() call.

        Returns:
            List of result lists, one per input query (same order).
        """
        tasks = [self.retrieve_async(q, **kwargs) for q in queries]
        results = await asyncio.gather(*tasks)
        return list(results)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def retrieve_from_source(
        self,
        query: str,
        source: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[RetrievedChunk]:
        """Retrieve only from a specific source document."""
        return self.retrieve(query, top_k=top_k, source=source, **kwargs)

    def retrieve_by_type(
        self,
        query: str,
        doc_type: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[RetrievedChunk]:
        """Retrieve only from a specific document type (e.g. 'pdf', 'html')."""
        return self.retrieve(query, top_k=top_k, doc_type=doc_type, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        doc_type: Optional[str],
        language: Optional[str],
        source: Optional[str],
        custom_filter: Optional[Filter],
    ) -> Optional[Filter]:
        """Combine any provided filters with AND logic."""
        parts: List[Filter] = []
        if doc_type:
            parts.append(filter_by_doc_type(doc_type))
        if language:
            parts.append(filter_by_language(language))
        if source:
            parts.append(filter_by_source(source))
        if custom_filter:
            parts.append(custom_filter)

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        # Combine multiple filters with AND
        combined = parts[0]
        for f in parts[1:]:
            combined = combined & f
        return combined

    @staticmethod
    def _deduplicate(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Remove duplicate results by (source, chunk_id), keeping the highest-scored one."""
        seen = {}
        for chunk in chunks:
            key = (chunk.source, chunk.chunk_id)
            if key not in seen or chunk.score > seen[key].score:
                seen[key] = chunk
        return list(seen.values())


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    w_client = WeaviateClient()
    embedder = GeminiEmbedder()
    retriever = VectorRetriever(w_client, embedder, default_top_k=5, default_distance_threshold=0.35)

    query = "What is UPI and how does it work in India?"
    results = retriever.retrieve(query)

    if results:
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] score={r.score} | source={r.source} | chunk={r.chunk_id}")
            print(f"     {r.content[:120]}...")
    else:
        print("[INFO] No results above threshold. Check that data is ingested.")

    # Multi-query async test
    async def multi_test():
        queries = [
            "FinTech growth in India",
            "RBI digital currency pilot",
            "NBFC regulatory framework",
        ]
        all_results = await retriever.retrieve_multi(queries, top_k=3)
        for q, res in zip(queries, all_results):
            print(f"\nQuery: '{q}' → {len(res)} results")
            for r in res:
                print(f"  score={r.score} | {r.content[:80]}...")

    asyncio.run(multi_test())

    w_client.close()