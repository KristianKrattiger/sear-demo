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
  seed                   control: swap seed.from for seed.to in a verified sentence and check the badge fires;
                         the same swap in the source chunk's sentence checks whether the gate would withhold it
<corpus>/narrator_controls.jsonl adds more controls per narrator probe, by type (interpretation, over-credit,
in-evidence swap, reversal, initial-name swap), each run through the narrator's labels and the gate (check, then
repair), with the checks that fired: entailment below tau, contradiction (narrator only), new names or numbers.

Per-probe detail is written to runs/<corpus>_eval.jsonl (and runs/<corpus>_narrator.jsonl).
--out puts them in another folder, and --set runs a config flag on without editing the config:

    python scripts/eval.py --narrator --out runs/<name> --set verifier.granularity=sentence
"""
import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from sear_demo.config import apply_overrides, load_config
from sear_demo.narrator import as_chunk, control as make_control, label, seeded
from sear_demo.corpus import load_probes
from sear_demo.pipeline import Sear
from sear_demo.schema import Claim
from sear_demo.segment import split_sentences

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="corpus", help="folder holding chunks.jsonl and probes.jsonl")
ap.add_argument("--narrator", action="store_true", help="also run <corpus>/narrator_probes.jsonl with the narrator on")
ap.add_argument("--out", default="runs", help="folder for the per-probe logs")
ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a config value (repeatable)")
args = ap.parse_args()

cfg = apply_overrides(load_config(), args.set)
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


def claim_row(c: Claim) -> dict:
    """A report claim for the log; a claim repair couldn't fix also logs repair's best candidate."""
    row = {"text": c.text, "cited": c.cited, "model_cited": c.model_cited, "score": round(c.score, 3), "verified": c.verified}
    if c.repair_best:
        row["repair_best"] = dict(c.repair_best, score=round(c.repair_best["score"], 3))
    return row


def gate_control(chunk_ids: list[str], find: str, to: str, also: list[str] = ()) -> Claim | None:
    """The first sentence of these chunks holding `find`, swapped for `to`, cited to its chunk and checked by the gate.
    `also` adds the first of these chunks holding each string to the citation, as control() does for handles.

    The generator kept every corpus value, so the gate never had to stop a swapped one; this asks whether it would."""
    for cid in chunk_ids:
        chunk = sear.by_id.get(cid)
        for sent in split_sentences(chunk.text) if chunk else []:
            if find.lower() in sent.lower():
                cited = [cid]
                for a in also:
                    extra = next((i for i in chunk_ids if i in sear.by_id and a.lower() in sear.by_id[i].text.lower()), None)
                    if extra is None:
                        return None
                    cited += [extra] if extra not in cited else []
                claim = Claim(re.sub(re.escape(find), to, sent, count=1, flags=re.I), cited)
                return sear.verifier.check([claim], sear.by_id)[0]
    return None


CONTROL_TYPES = ("swap", "initial-name swap", "in-evidence swap", "reversal", "over-credit", "interpretation")
control_tally = defaultdict(lambda: {"applied": 0, "badged": 0, "nar": Counter(), "missing": 0,
                                     "gate_applied": 0, "withheld": 0, "gate": Counter(), "slipped": []})


def fired(c: Claim, narrator: bool) -> list[str]:
    """The checks a control fails: entailment below tau, contradiction (the narrator's labels only), new names or numbers."""
    checks = [("entailment", c.score < sear.verifier.tau), ("contradiction", narrator and bool(c.contradicts)),
              ("new terms", bool(c.new_terms))]
    return [k for k, on in checks if on]


def run_control(r, kind: str, cid: str, find: str, to: str, also: list[str] = (), gate: dict | None = None) -> dict:
    """One control slip, labelled as a narrator sentence and, in its source chunk's wording, put through the gate.

    The gate side runs check and then repair, the gate's whole path, since repair could re-cite a slip to a chunk
    that passes it. Contradiction is logged on the gate side too, but the gate does not use it."""
    t, v = control_tally[kind], sear.verifier
    row = {"id": cid, "type": kind, "narrator": None, "gate": None}
    s = make_control(r, find, to, also)
    if s is None:
        t["missing"] += 1
    else:
        label(s, r.handles, sear.by_id, v, sear.cfg["narrator"]["contradict_tau"])
        sources = [as_chunk(h, r.handles[h], sear.by_id) for h in s.cited]
        f = fired(s, narrator=True)
        t["applied"] += 1
        t["badged"] += bool(f)
        t["nar"].update(f)
        t["slipped"] += [] if f else [f"{cid} (narrator)"]
        row["narrator"] = {"text": s.text, "cited": s.cited, "label": s.label, "score": round(s.score, 3),
                           "contradiction": round(v.contradiction(v.joint_premise(sources), s.text), 3),
                           "contradicts": s.contradicts, "new_terms": s.new_terms, "fired": f}
    chunk_ids = r.handles[s.cited[0]].cited if s else [c.id for c in r.retrieved]
    g = gate_control(chunk_ids, (gate or {}).get("find", find), (gate or {}).get("to", to), also)
    if g is not None:
        checked = {"score": round(g.score, 3), "new_terms": g.new_terms, "fired": fired(g, narrator=False)}
        if not g.verified:
            v.repair([g], r.retrieved)
        f = [] if g.verified else checked["fired"]
        t["gate_applied"] += 1
        t["withheld"] += bool(f)
        t["gate"].update(f)
        t["slipped"] += [] if f else [f"{cid} (gate{', after repair' if g.model_cited else ''})"]
        row["gate"] = {"text": g.text, "cited": g.cited, "model_cited": g.model_cited, "verified": g.verified, "check": checked,
                       "contradiction, not a gate check": round(v.contradiction(v.joint_premise([sear.by_id[i] for i in g.cited]), g.text), 3)}
    return row


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
        "claims": [claim_row(c) for c in r.claims],
    })

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
with open(out / f"{corpus.name}_eval.jsonl", "w", encoding="utf-8") as f:
    for row in log:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
dirty = " (uncommitted changes)" if git("status", "--porcelain", "--untracked-files=no") else ""
print(f"\ncommit {git('rev-parse', '--short', 'HEAD') or 'unknown'}{dirty}; overrides: {', '.join(args.set) or 'none'}; logs: {out}")
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
    controls = defaultdict(list)
    for c in load_probes(corpus / "narrator_controls.jsonl") if (corpus / "narrator_controls.jsonl").exists() else []:
        controls[c["probe"]].append(c)
    unbadged = sentences = supported = uncited = chained = unbacked = 0
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
        # source chains: cited handles whose sentence no cited chunk or pair entails fall back to every chunk
        chained += len(nar.sources)
        unbacked += len(nar.unbacked)
        # control: a known slip (seed.from swapped for seed.to in a verified sentence) must get a badge
        control = None
        if p.get("seed"):
            s = seeded(r, p["seed"]["from"], p["seed"]["to"])
            if s is not None:
                label(s, r.handles, sear.by_id, sear.verifier, sear.cfg["narrator"]["contradict_tau"])
                control = {"text": s.text, "cited": s.cited, "label": s.label, "score": round(s.score, 3), "new_terms": s.new_terms}
                score("seeded slip badged (control)", s.label != "supported", ntally)
                g = gate_control(r.handles[s.cited[0]].cited, p["seed"]["from"], p["seed"]["to"])
                if g is not None:
                    control["gate"] = {"text": g.text, "cited": g.cited, "score": round(g.score, 3), "verified": g.verified, "new_terms": g.new_terms}
                    score("gate withheld the seeded swap (control)", not g.verified, ntally)
                    score("gate withheld the seeded swap, NLI score alone", g.score < sear.verifier.tau, ntally)
        rows = [run_control(r, "swap", f"{p['id']}-seed", p["seed"]["from"], p["seed"]["to"])] if p.get("seed") else []
        rows += [run_control(r, c["type"], c["id"], c["find"], c["to"], c.get("also", []), c.get("gate"))
                 for c in controls.get(p["id"], [])]
        score("narrator ran", not nar.skipped, ntally)
        if not nar.skipped:  # a skipped narrator keeps nothing and slips nothing; score only the runs
            score("narrator kept the corpus value (runs)", kept, ntally)
            score("narrator avoided the slip, raw (runs)", not slip_raw, ntally)
        nlog.append({
            "id": p["id"], "question": p["question"], "keep": p["keep"], "slip_regex": p["slip_regex"], "report_failed": r.failed,
            "claims": [claim_row(c) for c in r.claims],
            "handles": {h: {"text": c.text, "cited": c.cited} for h, c in r.handles.items()},
            "skipped": nar.skipped, "unknown_ids": nar.unknown_ids,
            "unbacked": nar.unbacked,
            "sentences": [{"text": s.text, "cited": s.cited, "chain": {h: [c.id for c in chunks] for h, _, chunks in r.chain(s.cited, nar.sources)},
                           "unbacked": [h for h in s.cited if h in nar.unbacked],
                           "label": s.label, "score": round(s.score, 3), "contradicts": s.contradicts, "new_terms": s.new_terms}
                          for s in nar.sentences],
            "slip_raw": slip_raw, "slip_unbadged": slip_unbadged, "kept": kept, "control": control, "controls": rows,
        })

    with open(out / f"{corpus.name}_narrator.jsonl", "w", encoding="utf-8") as f:
        for row in nlog:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{corpus}: {len(nprobes)} narrator probes")
    print(f"{'narrator slips shown without a badge':<45} {unbadged}/{len(nprobes)}")
    print(f"{'summary sentences supported':<45} {supported}/{sentences}")
    print(f"{'summary sentences uncited or unknown handle':<45} {uncited}/{sentences}")
    print(f"{'source chains with no backing chunk or pair':<45} {unbacked}/{chained} cited handles")
    for name, (ok, n) in sorted(ntally.items()):
        print(f"{name:<45} {ok}/{n} ({ok / n:.0%})")

    # one line per control type: how many were caught, and by which check (a control can fail more than one)
    print("\ncontrols by type: narrator badged (checks that fired) | gate withheld after check and repair (checks that fired)")
    checks = lambda counter: ", ".join(f"{k} {counter[k]}" for k in ("entailment", "contradiction", "new terms") if counter[k]) or "none"
    for kind in sorted(control_tally, key=CONTROL_TYPES.index):
        t = control_tally[kind]
        line = (f"narrator {t['badged']}/{t['applied']} ({checks(t['nar'])}) | gate {t['withheld']}/{t['gate_applied']} ({checks(t['gate'])})"
                + (f"; did not apply {t['missing']}" if t["missing"] else "") + (f"; slipped: {', '.join(t['slipped'])}" if t["slipped"] else ""))
        print(f"{'control: ' + kind:<45} {line}")
