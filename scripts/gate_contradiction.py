"""What would a contradiction check in the gate catch, and what would it wrongly withhold? Measured, not applied.

    python scripts/gate_contradiction.py --runs runs/<name>     # reads <corpus>_eval.jsonl and <corpus>_narrator.jsonl

The narrator badges a summary sentence when its cited claims, joined, contradict it above contradict_tau;
the gate has no such check. Max-over-windows entailment lets a window that omits a fact "entail" the fact's
negation (a polarity control, 'not directed by Ridley Scott', passed at 0.978 while the window stating the
director contradicted it at 1.000). This applies the narrator's rule to the gate's claims, with the same
premise (the cited chunks joined, with film prefixes) and the same contradict_tau, and reports:
  - verified claims it would withhold: every one is a false positive unless the claim is wrong;
  - gate-side controls (from the narrator log) that slipped the gate and that it would catch.
"""
import argparse
import json
from pathlib import Path

from sear_demo.config import load_config
from sear_demo.corpus import load_chunks
from sear_demo.verify import Verifier

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
args = ap.parse_args()
cfg = load_config()
tau = cfg["narrator"]["contradict_tau"]
v = Verifier(cfg["verifier"]["model"], cfg["verifier"]["tau"])

for name, folder in (("corpus", "corpus"), ("poisoned", "corpus/poisoned")):
    log = Path(args.runs) / f"{name}_eval.jsonl"
    if not log.exists():
        continue
    by_id = {c.id: c for c in load_chunks(Path(folder) / "chunks.jsonl")}
    rows = [json.loads(line) for line in log.open(encoding="utf-8")]
    nlog = Path(args.runs) / f"{name}_narrator.jsonl"
    rows += [dict(r, probe=r["id"]) for r in map(json.loads, nlog.open(encoding="utf-8"))] if nlog.exists() else []
    verified = [(r["probe"], c) for r in rows for c in r["claims"] if c["verified"]]
    hits = []
    for probe, c in verified:
        score = v.contradiction(v.joint_premise([by_id[i] for i in c["cited"] if i in by_id]), c.get("checked") or c["text"])
        if score > tau:
            hits.append((probe, round(score, 3), c["text"]))
    print(f"{log.parent.name}/{name}: verified claims (probes and narrator reports) a gate contradiction check "
          f"(> {tau}) would withhold: {len(hits)}/{len(verified)}")
    for h in hits:
        print(f"  {h[0]} contradiction {h[1]}: {h[2]}")
    controls = [c for r in rows for c in r.get("controls", []) if c.get("gate")]
    slipped = [c for c in controls if c["gate"]["verified"]]
    caught = [c for c in slipped if c["gate"]["contradiction, not a gate check"] > tau]
    print(f"  gate-side controls that slipped the gate: {len(slipped)}/{len(controls)}; it would catch "
          f"{len(caught)}: {', '.join(c['id'] for c in caught) or 'none'}")
    for c in slipped:
        print(f"    {c['id']}: contradiction {c['gate']['contradiction, not a gate check']}")
