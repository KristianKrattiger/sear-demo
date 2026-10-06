from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .schema import Chunk


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: str, index_dir: str | Path):
        self.chunks = chunks
        self.model = SentenceTransformer(embedder, device="cpu")  # keep the GPU for the generator
        index_path = Path(index_dir) / "chunks.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            self.index = self.build(index_path)

    def build(self, index_path: Path):
        vecs = self.model.encode([c.text for c in self.chunks], normalize_embeddings=True)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(np.asarray(vecs, dtype="float32"))
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        return index

    def search(self, query: str, k: int) -> list[Chunk]:
        q = self.model.encode([query], normalize_embeddings=True)
        _, idx = self.index.search(np.asarray(q, dtype="float32"), k)
        return [self.chunks[i] for i in idx[0] if i >= 0]
