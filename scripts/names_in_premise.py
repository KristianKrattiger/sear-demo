"""What if a claim's names had to be in the premise that entails it? Measured, not applied.

    python scripts/names_in_premise.py --runs runs/<name>

The gate runs its two checks on different evidence: the entailment score is the best over many premises
(windows of each cited chunk, the joint premise, the focused joint premise), while the names check looks at
all the cited chunks joined. So a name can come from one chunk and the support from another. The
n04-initial control shows it: the check rejected 'Sam shoots the German Major Strasser ...' (Sam is not in
casa-c02), then repair re-cited it to casa-c02 + casa-c03, where casa-c03 mentions 'the pianist Sam' and a
casa-c02 window that says Rick shoots Strasser supplied the entailment.

The stricter rule: a claim verifies only if one premise both entails it (>= tau) and mentions every name and
number in it (film prefixes count). This replays every verified claim in a run's logs (would any be
wrongly dropped?) and every gate-side control that slipped (would it be caught?), over the citation each
ended with, so the repaired control is judged on the pair repair chose.
"""
import argparse
import json
from pathlib import Path

from sear_demo.config import load_config
from sear_demo.corpus import load_chunks
from sear_demo.verify import Verifier, corpus_names, new_terms

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
args = ap.parse_args()
cfg = load_config()
v = Verifier(cfg["verifier"]["model"], cfg["verifier"]["tau"])


def strict(text: str, sources) -> tuple[bool, float]:
    """Passes the stricter rule, and the best entailment among premises that name everything."""
    premises = [p for s in sources for p in v.premises(s)]
    if len(sources) > 1:
        premises += [v.joint_premise(sources), v.focused_joint_premise(sources, text)]
    named = [p for p in premises if not new_terms(text, p, v.names)]
    best = max((v.entailment(p, text) for p in named), default=0.0)
    return best >= v.tau, best


for name, folder in (("corpus", "corpus"), ("poisoned", "corpus/poisoned"), ("conflict", "corpus/conflict")):
    log = Path(args.runs) / f"{name}_eval.jsonl"
    if not log.exists():
        continue
    chunks = load_chunks(Path(folder) / "chunks.jsonl")
    by_id, v.names = {c.id: c for c in chunks}, corpus_names(c.text for c in chunks)
    rows = [json.loads(line) for line in log.open(encoding="utf-8")]
    nlog = Path(args.runs) / f"{name}_narrator.jsonl"
    rows += [dict(r, probe=r["id"]) for r in map(json.loads, nlog.open(encoding="utf-8"))] if nlog.exists() else []
    verified = [(r["probe"], c) for r in rows for c in r["claims"] if c["verified"]]
    dropped = []
    for probe, c in verified:
        ok, best = strict(c.get("checked") or c["text"], [by_id[i] for i in c["cited"] if i in by_id])
        if not ok:
            dropped.append((probe, round(best, 3), c["text"]))
    print(f"{log.parent.name}/{name}: verified claims the stricter rule would drop: {len(dropped)}/{len(verified)}")
    for d in dropped:
        print(f"  {d[0]} best entailment among premises naming everything {d[1]}: {d[2]}")
    slipped = [c for r in rows for c in r.get("controls", []) if c.get("gate") and c["gate"]["verified"]]
    caught = []
    for c in slipped:
        ok, best = strict(c["gate"]["text"], [by_id[i] for i in c["gate"]["cited"]])
        print(f"  slipped control {c['id']} cited {'+'.join(c['gate']['cited'])}: "
              f"{'still passes' if ok else 'caught'} ({best:.3f})")
        caught += [] if ok else [c["id"]]
    print(f"  gate-side controls that slipped the gate: {len(slipped)}; it would catch {len(caught)}: {', '.join(caught) or 'none'}")
