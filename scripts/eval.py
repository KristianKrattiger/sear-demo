"""Run a probe set and print SEAR's headline metrics (ENG 01 §6).

    python scripts/eval.py                                     # corpus/
    python scripts/eval.py --corpus corpus/poisoned            # the poison fork
    python scripts/eval.py --narrator                          # also run corpus/narrator_probes.jsonl
    python scripts/eval.py --corpus corpus/poisoned --narrator # same for the poison fork

Probe layers:
  realism, multi_hop     answer correctly from the corpus
  counterfactual         answer from the corpus even though it contradicts the model's memory
  poisoned               same, but the corpus is silently wrong (the governance limit)
  out_of_corpus,
  near_miss              emit the failure state; near_miss has neighbouring facts in the corpus
  overreach              never emit a claim matching forbid_regex (the source doesn't support it)

Narrator probes (--narrator, <corpus>/narrator_probes.jsonl):
  keep                   the summary keeps this corpus value in a sentence labelled supported
  slip_regex             never shown as supported; a contradicts/unsupported badge catches the slip
  seed                   control: swap seed.from for seed.to in a verified sentence and check the badge fires

Per-probe detail is written to runs/<corpus>_eval.jsonl (and runs/<corpus>_narrator.jsonl).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sear_demo.config import load_config
from sear_demo.narrator import label, seeded
from sear_demo.corpus import load_probes
from sear_demo.pipeline import Sear

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="corpus", help="folder holding chunks.jsonl and probes.jsonl")
ap.add_argument("--narrator", action="store_true", help="also run <corpus>/narrator_probes.jsonl with the narrator on")
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
emitted = unverified = recited = 0
log = []
SHOULD_REFUSE = ("out_of_corpus", "near_miss")
refusals = []  # (probe id, layer, who): the generator wrote NOT SUPPORTED, or the gate withheld its claims


def score(name: str, ok: bool, t=tally):
    t[name][0] += bool(ok)
    t[name][1] += 1


for p in probes:
    r = sear.ask(p["question"], narrator=False).report
    kept = "" if r.failed else " ".join(c.text for c in r.verified_claims)
    emitted += len(r.claims)
    unverified += sum(not c.verified for c in r.claims)
    recited += sum(c.model_cited is not None for c in r.claims)
    layer = p["layer"]
    if r.failed:
        refusals.append((p["id"], layer, "gate" if r.claims else "generator"))

    if layer in SHOULD_REFUSE:
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
        "claims": [{"text": c.text, "cited": c.cited, "model_cited": c.model_cited, "score": round(c.score, 3), "verified": c.verified} for c in r.claims],
    })

Path("runs").mkdir(exist_ok=True)
with open(f"runs/{corpus.name}_eval.jsonl", "w", encoding="utf-8") as f:
    for row in log:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\n{corpus}: {len(probes)} probes")
print(f"{'claims flagged unsupported by the gate':<45} {unverified}/{emitted}")
print(f"{'claims re-cited by the verifier':<45} {recited}/{emitted}")
for name, (ok, n) in sorted(tally.items()):
    print(f"{name:<45} {ok}/{n} ({ok / n:.0%})")

# ENG 01 §6 failure-state precision/recall. Only out_of_corpus and near_miss should refuse; a refusal
# anywhere else (an overreach probe with a supportable partial answer, a missed multi-hop) is over-refusal.
should = sum(p["layer"] in SHOULD_REFUSE for p in probes)
right = sum(layer in SHOULD_REFUSE for _, layer, _ in refusals)
by_gate = [pid for pid, _, who in refusals if who == "gate"]
print(f"{'failure states':<45} {len(refusals)}/{len(probes)} "
      f"(generator refused {len(refusals) - len(by_gate)}, gate withheld {len(by_gate)}: {', '.join(by_gate) or 'none'})")
if should:
    print(f"{'failure-state recall':<45} {right}/{should} ({right / should:.0%})")
if refusals:
    over = ", ".join(f"{pid} ({layer})" for pid, layer, _ in refusals if layer not in SHOULD_REFUSE) or "none"
    print(f"{'failure-state precision':<45} {right}/{len(refusals)} ({right / len(refusals):.0%}); over-refused: {over}")

narrator_path = corpus / "narrator_probes.jsonl"
if args.narrator and not narrator_path.exists():
    print(f"\n{corpus}: no narrator_probes.jsonl, narrator probes skipped")
elif args.narrator:
    nprobes, ntally, nlog = load_probes(narrator_path), defaultdict(lambda: [0, 0]), []
    unbadged = sentences = supported = uncited = 0
    for p in nprobes:
        res = sear.ask(p["question"], narrator=True)
        r, nar = res.report, res.narration
        rx = re.compile(p["slip_regex"], re.I)
        shown = [s for s in nar.sentences if s.label == "supported"]
        slip_raw = bool(rx.search(" ".join(s.text for s in nar.sentences)))
        slip_unbadged = any(rx.search(s.text) for s in shown)
        kept = any(hits(p["keep"], s.text) for s in shown)
        unbadged += slip_unbadged
        sentences += len(nar.sentences)
        supported += len(shown)
        # cited keeps only valid handles: a sentence that cited only unknown ones is left with none
        uncited += sum(s.label == "uncited" or not s.cited for s in nar.sentences)
        # control: a known slip (seed.from swapped for seed.to in a verified sentence) must get a badge
        control = None
        if p.get("seed"):
            s = seeded(r, p["seed"]["from"], p["seed"]["to"])
            if s is not None:
                label(s, r.handles, sear.by_id, sear.verifier, sear.cfg["narrator"]["contradict_tau"])
                control = {"text": s.text, "cited": s.cited, "label": s.label, "score": round(s.score, 3)}
                score("seeded slip badged (control)", s.label != "supported", ntally)
        score("narrator ran", not nar.skipped, ntally)
        if not nar.skipped:  # a skipped narrator keeps nothing and slips nothing; score only the runs
            score("narrator kept the corpus value (runs)", kept, ntally)
            score("narrator avoided the slip, raw (runs)", not slip_raw, ntally)
        nlog.append({
            "id": p["id"], "question": p["question"], "keep": p["keep"], "slip_regex": p["slip_regex"], "report_failed": r.failed,
            "handles": {h: {"text": c.text, "cited": c.cited} for h, c in r.handles.items()},
            "skipped": nar.skipped, "unknown_ids": nar.unknown_ids,
            "sentences": [{"text": s.text, "cited": s.cited, "chain": {h: [c.id for c in chunks] for h, _, chunks in r.chain(s.cited, nar.sources)},
                           "label": s.label, "score": round(s.score, 3), "contradicts": s.contradicts} for s in nar.sentences],
            "slip_raw": slip_raw, "slip_unbadged": slip_unbadged, "kept": kept, "control": control,
        })

    with open(f"runs/{corpus.name}_narrator.jsonl", "w", encoding="utf-8") as f:
        for row in nlog:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{corpus}: {len(nprobes)} narrator probes")
    print(f"{'narrator slips shown without a badge':<45} {unbadged}/{len(nprobes)}")
    print(f"{'summary sentences supported':<45} {supported}/{sentences}")
    print(f"{'summary sentences uncited or unknown handle':<45} {uncited}/{sentences}")
    for name, (ok, n) in sorted(ntally.items()):
        print(f"{name:<45} {ok}/{n} ({ok / n:.0%})")
