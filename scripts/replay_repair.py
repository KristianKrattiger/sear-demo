"""Replay citation repair for one logged claim and show every candidate's score.

    python scripts/replay_repair.py p15                                   # runs/corpus_eval.jsonl
    python scripts/replay_repair.py p15 --log runs/<dir>/corpus_eval.jsonl

A dropped claim is either a search miss (repair never tried the chunks that support it) or a
verifier miss (it tried them and the score fell below tau, or a name or number was missing).
The eval log keeps only the final verdict, so this re-runs Verifier.repair's candidate list in
its order and prints, for each candidate: its chunks, the best entailment, the premise that gave
it, and the names and numbers the candidate lacks. Nothing here changes the pipeline.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

from sear_demo.config import load_config
from sear_demo.corpus import load_chunks
from sear_demo.verify import Verifier, new_terms

ap = argparse.ArgumentParser()
ap.add_argument("probe")
ap.add_argument("--log", default="runs/corpus_eval.jsonl")
ap.add_argument("--corpus", default="corpus", help="folder holding the chunks the log was made on")
ap.add_argument("--pair-pool", type=int, default=6, help="Verifier.repair's pair_pool")
ap.add_argument("--detail", help="comma-separated chunk IDs: print every premise's score for that candidate")
args = ap.parse_args()

cfg = load_config()
v = Verifier(cfg["verifier"]["model"], cfg["verifier"]["tau"])
by_id = {c.id: c for c in load_chunks(Path(args.corpus) / "chunks.jsonl")}
row = next(json.loads(l) for l in open(args.log, encoding="utf-8") if json.loads(l)["probe"] == args.probe)
retrieved = [by_id[i] for i in row["retrieved"]]
print(f"{row['probe']} ({row['layer']}): {row['question']}\nretrieved: {', '.join(row['retrieved'])}\n")


def candidates(own):
    """Verifier.repair's candidates, in its order, de-duplicated the same way."""
    out = [[c] for c in retrieved]
    out += [own + [c] for c in retrieved if c not in own]
    out += [list(p) for p in combinations(retrieved[:args.pair_pool], 2)]
    seen, kept = set(), []
    for cand in out:
        key = frozenset(c.id for c in cand)
        if key not in seen:
            seen.add(key)
            kept.append(cand)
    return kept


for claim in row["claims"]:
    if claim["verified"]:
        continue
    text, own = claim["text"], [by_id[i] for i in claim["cited"] if i in by_id]
    print(f"claim: {text}\nmodel cited: {claim['cited']}  logged score: {claim['score']}")
    if not own:
        print("  uncited: repair never runs on it\n")
        continue
    rows = []
    for n, cand in enumerate(candidates(own), 1):
        premises = [p for s in cand for p in v.premises(s)]
        if len(cand) > 1:
            premises += [v.joint_premise(cand), v.focused_joint_premise(cand, text)]
        scored = [(v.entailment(p, text), p) for p in premises]
        best, premise = max(scored)
        terms = new_terms(text, v.joint_premise(cand))
        rows.append((n, [c.id for c in cand], best, terms, premise))
    rows.sort(key=lambda r: -r[2])
    print(f"  {len(rows)} candidates tried; passing needs score >= {v.tau} and no missing names or numbers")
    shown = rows[:8] + [r for r in rows[8:] if not r[3]]  # the top scores, and every candidate the names check passes
    for n, ids, best, terms, premise in shown:
        verdict = "PASS" if best >= v.tau and not terms else "fail"
        print(f"  #{n:<3} {'+'.join(ids):<24} entailment {best:.3f}  missing: {', '.join(terms) or 'none'}  {verdict}")
        print(f"        best premise: {premise[:240]}")
    if args.detail:
        cand = [by_id[i] for i in args.detail.split(",")]
        named = [(p, "window") for s in cand for p in v.premises(s)]
        if len(cand) > 1:
            named += [(v.joint_premise(cand), "joint"), (v.focused_joint_premise(cand, text), "focused joint")]
        print(f"\n  every premise for {args.detail}:")
        for score, kind, p in sorted(((v.entailment(p, text), kind, p) for p, kind in named), reverse=True):
            print(f"    {score:.3f}  {kind:<13} {p[:200]}")
    print()
