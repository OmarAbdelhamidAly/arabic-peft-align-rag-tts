"""
scripts/ingest_documents.py
-----------------------------
TODO: Standalone script to bulk-ingest Arabic medical documents into Qdrant.

Run this ONCE (or whenever you add new documents) to populate the vector store.

Usage:
    python scripts/ingest_documents.py --source ./data/documents --collection arabic_medical

Steps to implement:
  1. Walk the source directory for PDF, TXT, DOCX files
  2. For each file:
     a. Parse text (PyMuPDF for PDF, plain read for TXT)
     b. Normalize Arabic text (optional: remove tashkeel)
     c. Split into chunks using RecursiveCharacterTextSplitter
     d. Add source metadata to each chunk
  3. Embed all chunks in batches (E5EmbeddingAdapter)
  4. Upsert into Qdrant (QdrantRetriever.add_documents)
  5. Log progress and final stats

Research questions:
  - Arabic text splitting: does LangChain's splitter handle RTL correctly?
  - Should we store the raw text or normalized text in Qdrant?
  - How to handle duplicate documents on re-ingest?
    → Use document hash as Qdrant point ID

Useful libraries: pymupdf (fitz), langchain-text-splitters, qdrant-client, tqdm
"""

import argparse
from pathlib import Path


def ingest_directory(source_dir: str, collection: str, qdrant_host: str) -> None:
    """
    TODO: Walk source_dir, parse all documents, embed, and upsert to Qdrant.
    """
    raise NotImplementedError("Implement document ingestion pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Arabic medical documents into Qdrant")
    parser.add_argument("--source",     default="./data/documents", help="Directory with source documents")
    parser.add_argument("--collection", default="arabic_medical",   help="Qdrant collection name")
    parser.add_argument("--qdrant",     default="localhost",        help="Qdrant host")
    args = parser.parse_args()

    ingest_directory(args.source, args.collection, args.qdrant)
