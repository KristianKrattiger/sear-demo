"""Run a probe set and print SEAR's headline metrics (ENG 01 §6).

    python scripts/eval.py                          # corpus/
    python scripts/eval.py --corpus corpus/poisoned # the poison fork

Probe layers:
  realism, multi_hop     answer correctly from the corpus
  counterfactual         answer from the corpus even though it contradicts the model's memory
  poisoned               same, but the corpus is silently wrong (the governance limit)
  out_of_corpus,
  near_miss              emit the failure state; near_miss has neighbouring facts in the corpus
  overreach              never emit a claim matching forbid_regex (the source doesn't support it)

Per-probe detail is written to runs/<corpus>_eval.jsonl.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sear_demo.config import load_config
from sear_demo.corpus import load_probes
from sear_demo.pipeline import Sear

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="corpus", help="folder holding chunks.jsonl and probes.jsonl")
args = ap.parse_args()

cfg = load_config()
corpus = Path(args.corpus)
index_dir = Path(cfg["corpus"]["index_dir"]) / corpus.name
cfg["corpus"].update(chunks=str(corpus / "chunks.jsonl"), probes=str(corpus / "probes.jsonl"), index_dir=str(index_dir))

sear = Sear(cfg)
sear.retriever.index = sear.retriever.build(index_dir / "chunks.faiss")  # always rebuild: corpus files change
probes = load_probes(cfg["corpus"]["probes"])


def hits(expect, text: str) -> bool:
    options = expect if isinstance(expect, list) else [expect]
    return any(o and o.lower() in text.lower() for o in options)


tally = defaultdict(lambda: [0, 0])
emitted = unverified = 0
log = []


def score(name: str, ok: bool):
    tally[name][0] += bool(ok)
    tally[name][1] += 1


for p in probes:
    r = sear.ask(p["question"], narrator=False).report
    kept = "" if r.failed else " ".join(c.text for c in r.verified_claims)
    emitted += len(r.claims)
    unverified += sum(not c.verified for c in r.claims)
    layer = p["layer"]

    if layer in ("out_of_corpus", "near_miss"):
        score(f"refused ({layer})", r.failed)
    elif layer == "overreach":
        rx = re.compile(p["forbid_regex"], re.I)
        # a refusal that echoes the question ("...why Ripley destroys...") is not overreach
        score("overreach: model avoided it (raw)", r.failed or not rx.search(r.raw))
        score("overreach: nothing unsupported emitted", r.failed or not rx.search(kept))
    else:
        score(f"answered correctly ({layer})", not r.failed and hits(p["expect"], kept))
        if layer in ("counterfactual", "poisoned"):
            score(f"model followed corpus, raw ({layer})", hits(p["expect"], r.raw))

    log.append({
        "probe": p["id"], "layer": layer, "question": p["question"], "expect": p.get("expect"),
        "failed": r.failed, "retrieved": [c.id for c in r.retrieved], "raw": r.raw,
        "claims": [{"text": c.text, "cited": c.cited, "score": round(c.score, 3), "verified": c.verified} for c in r.claims],
    })

Path("runs").mkdir(exist_ok=True)
with open(f"runs/{corpus.name}_eval.jsonl", "w", encoding="utf-8") as f:
    for row in log:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\n{corpus}: {len(probes)} probes")
print(f"{'claims flagged unsupported by the gate':<45} {unverified}/{emitted}")
for name, (ok, n) in sorted(tally.items()):
    print(f"{name:<45} {ok}/{n} ({ok / n:.0%})")
