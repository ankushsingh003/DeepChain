"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Data Loading Logic
"""

import os
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_documents(self) -> List[Document]:
        """Loads all documents from the specified directory."""
        print(f"[*] Loading documents from {self.data_path}...")
        
        # Support for .txt
        text_loader = DirectoryLoader(self.data_path, glob="**/*.txt", loader_cls=TextLoader)
        # Support for .pdf
        pdf_loader = DirectoryLoader(self.data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        
        docs = text_loader.load() + pdf_loader.load()
        print(f"[+] Loaded {len(docs)} documents.")
        return docs

if __name__ == "__main__":
    # Test loader
    loader = DocumentLoader("data/sample_docs")
    documents = loader.load_documents()
    for doc in documents:
        print(f"Source: {doc.metadata.get('source')}, Content snippet: {doc.page_content[:100]}...")
