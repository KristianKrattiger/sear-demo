"""Sentence-level verification, replayed on logged reports under each antecedent remedy.

    python scripts/replay_sentences.py --runs runs/<name>            # a claim-mode run with --narrator
    python scripts/replay_sentences.py --runs runs/<name> --corpus corpus/poisoned

The 27 probes' answers are single sentences, so sentence mode barely touches them; the antecedent
problem shows in the narrator probes' multi-sentence reports. This takes each logged report as the
generator wrote it (its claims with the model's own citations), re-runs retrieval (deterministic),
and verifies it the way verifier.granularity: sentence would, once per sentence_context. Same text
for every remedy, no generation. For each remedy it prints the sentences verified and dropped, the
reports that would fail the gate, and every dropped sentence with what was scored.
"""
import argparse
import json
from pathlib import Path

from sear_demo.config import load_config
from sear_demo.corpus import load_chunks
from sear_demo.gate import gate
from sear_demo.retrieve import Retriever
from sear_demo.schema import Claim
from sear_demo.segment import sentence_claims
from sear_demo.verify import Verifier, corpus_names

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--corpus", default="corpus")
args = ap.parse_args()

cfg = load_config()
corpus = Path(args.corpus)
chunks = load_chunks(corpus / "chunks.jsonl")
v = Verifier(cfg["verifier"]["model"], cfg["verifier"]["tau"])
v.names = corpus_names(c.text for c in chunks)
retriever = Retriever(chunks, cfg["retrieval"]["embedder"], Path(cfg["corpus"]["index_dir"]) / corpus.name)
retriever.index = retriever.build(Path(cfg["corpus"]["index_dir"]) / corpus.name / "chunks.faiss")
phi = cfg["gate"]["phi"]
rows = [json.loads(line) for line in (Path(args.runs) / f"{corpus.name}_narrator.jsonl").open(encoding="utf-8")]

print(f"{args.runs} / {corpus.name}: {len(rows)} narrator-probe reports, replayed in sentence mode\n")
for context in ("none", "film", "previous", "resolve"):
    total = verified = failed = 0
    dropped = []
    for row in rows:
        retrieved = retriever.search(row["question"], cfg["retrieval"]["top_k"])
        films = {c.id: c.meta.get("film") for c in retrieved}
        written = [Claim(c["text"], list(c["model_cited"] or c["cited"])) for c in row["claims"]]
        claims = sentence_claims(written, context, films)
        v.check(claims, {c.id: c for c in retrieved})
        v.repair(claims, retrieved)
        total += len(claims)
        verified += sum(c.verified for c in claims)
        failed += not gate(claims, phi)
        dropped += [(row["id"], c) for c in claims if not c.verified]
    print(f"sentence_context={context:<9} sentences verified {verified}/{total}, dropped {total - verified}, "
          f"reports that would fail the gate {failed}/{len(rows)}")
    for pid, c in dropped:
        scored = f" | scored: {c.checked}" if c.checked else ""
        terms = f" | missing: {', '.join(c.new_terms)}" if c.new_terms else ""
        print(f"    {pid} {c.score:.3f} {c.text}{scored}{terms}")
    print()
