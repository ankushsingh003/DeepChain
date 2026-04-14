"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Semantic Chunking Strategy
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller chunks for vector and graph processing."""
        print(f"[*] Splitting {len(documents)} documents into chunks...")
        chunks = self.splitter.split_documents(documents)
        print(f"[+] Created {len(chunks)} chunks.")
        return chunks

if __name__ == "__main__":
    from langchain_core.documents import Document
    
    # Test sample
    test_docs = [Document(page_content="DeepChain-Hybrid-RAG is a powerful system. " * 50)]
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_documents(test_docs)
    print(f"[TEST] Chunk count: {len(chunks)}")
    print(f"[TEST] First chunk: {chunks[0].page_content}")
