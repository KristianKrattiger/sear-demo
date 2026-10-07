import json
from pathlib import Path

from .schema import Chunk


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            chunks.append(Chunk(
                id=row["id"],
                text=row["text"],
                head=row.get("head", ""),
                layer=row.get("layer", "realism"),
                meta={k: v for k, v in row.items() if k not in {"id", "text", "head", "layer"}},
            ))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk IDs in corpus; IDs must be stable and unique")
    return chunks


def corpora(root: str | Path = "corpus") -> dict[str, Path]:
    """Corpus folders by name: root itself, then each subfolder holding a chunks.jsonl (the forks)."""
    root = Path(root)
    found = {root.name: root} if (root / "chunks.jsonl").exists() else {}
    found.update({d.name: d for d in sorted(root.iterdir()) if d.is_dir() and (d / "chunks.jsonl").exists()})
    return found


def load_probes(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
