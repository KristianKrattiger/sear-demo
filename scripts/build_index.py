"""Embed corpus/chunks.jsonl into index/chunks.faiss."""
import shutil
from pathlib import Path

from sear_demo.config import load_config
from sear_demo.corpus import load_chunks
from sear_demo.retrieve import Retriever

cfg = load_config()
shutil.rmtree(Path(cfg["corpus"]["index_dir"]), ignore_errors=True)
chunks = load_chunks(cfg["corpus"]["chunks"])
Retriever(chunks, cfg["retrieval"]["embedder"], cfg["corpus"]["index_dir"])
print(f"indexed {len(chunks)} chunks")
