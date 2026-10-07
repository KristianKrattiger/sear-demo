"""Is the NLI verifier's input cut at 512 tokens, and how well does it read a long premise?

    python scripts/check_truncation.py                 # after scripts/eval.py --narrator, which writes runs/corpus_*.jsonl
    python scripts/check_truncation.py --runs runs/x   # logs from eval.py --out runs/x

1. Rebuilds every NLI input the gate scored in the last eval (each claim against its cited chunks'
   windows, joint premise and focused joint premise) and counts tokens, for the probes and for the
   narrator probes' reports. The long inputs come from the overview questions, where one paragraph
   cites every chunk of a film. For the longest it prints the tokens the model actually receives and
   the score with and without a forced cut at 512: received equal to the full count means nothing was cut.
2. Scores one fact against its own chunk and against the four Alien chunks joined, which is why the
   gate scores short windows and takes the best, rather than trusting the joint premise alone.
"""
import argparse
import json
from pathlib import Path

from sear_demo.config import load_config
from sear_demo.corpus import load_chunks
from sear_demo.verify import Verifier

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs", help="folder holding the eval logs")
args = ap.parse_args()

cfg = load_config()
v = Verifier(cfg["verifier"]["model"], cfg["verifier"]["tau"])
chunks = {c.id: c for c in load_chunks(cfg["corpus"]["chunks"])}
name = Path(cfg["corpus"]["chunks"]).parent.name
logs = [Path(args.runs) / f"{name}_eval.jsonl", Path(args.runs) / f"{name}_narrator.jsonl"]

count = lambda premise, claim: len(v.nli.tokenizer(premise, claim)["input_ids"])
received = lambda premise, claim: v.nli.preprocess({"text": premise, "text_pair": claim})["input_ids"].shape[-1]
entail = lambda premise, claim, **cut: next(s["score"] for s in v.nli({"text": premise, "text_pair": claim}, **cut)
                                            if s["label"].lower().startswith("entail"))


def inputs(claim: str, sources) -> list[tuple[str, str]]:
    """The (premise, claim) pairs Verifier.support scores."""
    premises = [p for s in sources for p in v.premises(s)]
    if len(sources) > 1:
        premises += [v.joint_premise(sources), v.focused_joint_premise(sources, claim)]
    return [(p, claim) for p in premises]


def claims(row: dict) -> list[dict]:
    """A logged report's claims. Older narrator logs keep only the handles: consecutive handles
    citing the same chunks are the sentences of one claim, so join them back."""
    if "claims" in row:
        return row["claims"]
    out = []
    for h in row.get("handles", {}).values():
        if out and out[-1]["cited"] == h["cited"]:
            out[-1]["text"] += " " + h["text"]
        else:
            out.append({"text": h["text"], "cited": list(h["cited"])})
    return out


pairs = []
for log in logs:
    if not log.exists():
        print(f"{log} not found: run scripts/eval.py --narrator first")
        continue
    found = []
    for line in log.open(encoding="utf-8"):
        for c in claims(json.loads(line)):
            for cited in (c["cited"], c.get("model_cited") or []):  # the gate scored the model's own citation first
                sources = [chunks[i] for i in cited if i in chunks]
                if sources:
                    found += inputs(c["text"], sources)
    sizes = [count(p, h) for p, h in found]
    if sizes:
        print(f"NLI inputs the gate scored for {log}: {len(found)}, over 512 tokens: {sum(n > 512 for n in sizes)}, "
              f"longest: {max(sizes)} tokens")
    pairs += found
if pairs:
    sizes = [count(p, h) for p, h in pairs]
    premise, claim = pairs[max(range(len(pairs)), key=sizes.__getitem__)]
else:
    print("No eval logs: using the four Alien chunks against a paragraph-long claim.")
    alien = [c for i, c in sorted(chunks.items()) if i.startswith("alien-")]
    premise, claim = v.joint_premise(alien), " ".join(c.text for c in alien[:2])

print(f"longest input: {count(premise, claim)} tokens, the model receives {received(premise, claim)} "
      f"(tokenizer max {v.nli.tokenizer.model_max_length}, "
      f"position_biased_input={getattr(v.nli.model.config, 'position_biased_input', None)})")
print(f"  entailment as the gate scores it: {entail(premise, claim):.3f}; "
      f"with the premise cut to 512: {entail(premise, claim, truncation='only_first', max_length=512):.3f}")

alien = [c for i, c in sorted(chunks.items()) if i.startswith("alien-")]
fact, own = "Alien was released in 1983.", next(c for c in alien if "1983" in c.text)
print(f"\n'{fact}' against {own.id}'s windows (best): {max(entail(p, fact) for p in v.premises(own)):.3f}; "
      f"against all {len(alien)} Alien chunks joined ({count(v.joint_premise(alien), fact)} tokens): "
      f"{entail(v.joint_premise(alien), fact):.3f}")
