# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: Gemini Embedding Wrapper
# """

# import os
# from typing import List
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()
# class GeminiEmbedder:
#     def __init__(self, model: str | None = None):
#         model = model or os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
#         self.embeddings = GoogleGenerativeAIEmbeddings(model=model)

#     def embed_query(self, text: str) -> List[float]:
#         """Generates embedding for a single query string."""
#         return self.embeddings.embed_query(text)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Generates embeddings for a list of document strings."""
#         return self.embeddings.embed_documents(texts)

#     def get_embedding_function(self):
#         """Returns the underlying LangChain embedding function."""
#         return self.embeddings

# if __name__ == "__main__":
#     # Test embedder
#     embedder = GeminiEmbedder()
#     vector = embedder.embed_query("FinTech growth in India")
#     print(f"[TEST] Vector length: {len(vector)}")
#     print(f"[TEST] First 5 dims: {vector[:5]}")












"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Gemini Embedding Wrapper — Production Grade

Fixes & Additions over v1:
  - Batched embedding with configurable batch_size to avoid API rate limits
  - Exponential backoff retry on transient API failures
  - Async concurrent embedding via asyncio + semaphore
  - Token-count estimation to guard against oversized inputs
  - Logging instead of bare prints
  - Type hints throughout
"""

import os
import time
import asyncio
import logging
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 100          # Gemini safe batch size per API call
DEFAULT_RETRY_ATTEMPTS = 3        # Max retries on transient failure
DEFAULT_RETRY_DELAY = 1.0         # Initial backoff in seconds (doubles each retry)
DEFAULT_ASYNC_CONCURRENCY = 5     # Max parallel embedding batches
MAX_CHARS_PER_TEXT = 25_000       # Rough guard — Gemini token limit proxy (~6k tokens)


class GeminiEmbedder:
    """
    Production-grade wrapper around Gemini text embeddings.

    Features:
        - Synchronous batched embedding with retry + backoff
        - Async concurrent batched embedding
        - Input length guard
        - Transparent progress logging
    """

    def __init__(
        self,
        model: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        async_concurrency: int = DEFAULT_ASYNC_CONCURRENCY,
    ):
        model = model or os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.async_concurrency = async_concurrency
        logger.info(f"[GeminiEmbedder] Initialized with model={model}, batch_size={batch_size}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _guard_input_length(self, texts: List[str]) -> List[str]:
        """Truncate texts that exceed the rough character cap to avoid API errors."""
        guarded = []
        for t in texts:
            if len(t) > MAX_CHARS_PER_TEXT:
                logger.warning(
                    f"[GeminiEmbedder] Text truncated from {len(t)} to {MAX_CHARS_PER_TEXT} chars."
                )
                guarded.append(t[:MAX_CHARS_PER_TEXT])
            else:
                guarded.append(t)
        return guarded

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """Embed a single batch with exponential backoff retry."""
        delay = self.retry_delay
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return self.embeddings.embed_documents(batch)
            except Exception as e:
                if attempt == self.retry_attempts:
                    logger.error(f"[GeminiEmbedder] Batch failed after {attempt} attempts: {e}")
                    raise
                logger.warning(
                    f"[GeminiEmbedder] Attempt {attempt} failed ({e}). Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2  # exponential backoff
        return []  # unreachable, satisfies type checker

    # ------------------------------------------------------------------
    # Public synchronous API
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        """Generates embedding for a single query string."""
        text = self._guard_input_length([text])[0]
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.
        Automatically splits into batches and retries on failure.
        """
        texts = self._guard_input_length(texts)
        all_vectors: List[List[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        logger.info(f"[GeminiEmbedder] Embedding {len(texts)} texts in {total_batches} batches...")

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch = texts[start:end]

            logger.info(f"[GeminiEmbedder] Batch {batch_idx + 1}/{total_batches} ({len(batch)} texts)...")
            vectors = self._embed_batch_with_retry(batch)
            all_vectors.extend(vectors)

            # Polite delay between batches to avoid rate-limit bursts
            if batch_idx < total_batches - 1:
                time.sleep(0.3)

        logger.info(f"[GeminiEmbedder] Done. Total vectors: {len(all_vectors)}")
        return all_vectors

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def _embed_batch_async(
        self,
        batch: List[str],
        semaphore: asyncio.Semaphore,
        batch_idx: int,
        total_batches: int,
    ) -> List[List[float]]:
        """Async wrapper that acquires semaphore before calling the sync embed."""
        async with semaphore:
            logger.info(f"[GeminiEmbedder][async] Batch {batch_idx + 1}/{total_batches}")
            loop = asyncio.get_event_loop()
            # Run blocking Gemini call in a thread pool
            return await loop.run_in_executor(
                None, self._embed_batch_with_retry, batch
            )

    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """
        Async concurrent embedding — significantly faster for large corpora.
        Concurrency is bounded by self.async_concurrency to avoid rate limiting.
        """
        texts = self._guard_input_length(texts)
        batches = [
            texts[i: i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        total_batches = len(batches)
        semaphore = asyncio.Semaphore(self.async_concurrency)

        logger.info(
            f"[GeminiEmbedder][async] Embedding {len(texts)} texts "
            f"in {total_batches} batches (concurrency={self.async_concurrency})..."
        )

        tasks = [
            self._embed_batch_async(batch, semaphore, idx, total_batches)
            for idx, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks)

        # Flatten: results is List[List[List[float]]]
        all_vectors: List[List[float]] = []
        for batch_vectors in results:
            all_vectors.extend(batch_vectors)

        logger.info(f"[GeminiEmbedder][async] Done. Total vectors: {len(all_vectors)}")
        return all_vectors

    def get_embedding_function(self):
        """Returns the underlying LangChain embedding function (for compatibility)."""
        return self.embeddings


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    embedder = GeminiEmbedder(batch_size=5)

    # Sync test
    texts = [f"Sample document number {i} about FinTech in India." for i in range(12)]
    vectors = embedder.embed_documents(texts)
    print(f"[TEST][sync] {len(vectors)} vectors, dim={len(vectors[0])}")

    # Single query
    qv = embedder.embed_query("What is UPI in India?")
    print(f"[TEST][query] dim={len(qv)}, first5={qv[:5]}")

    # Async test
    async def async_test():
        vecs = await embedder.embed_documents_async(texts)
        print(f"[TEST][async] {len(vecs)} vectors, dim={len(vecs[0])}")

    asyncio.run(async_test())