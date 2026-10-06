"""Run the probe set and print the Stage 1 headline metrics (ENG 01 §6)."""
from sear_demo.config import load_config
from sear_demo.corpus import load_probes
from sear_demo.pipeline import Sear

cfg = load_config()
sear = Sear(cfg)
probes = load_probes(cfg["corpus"]["probes"])

emitted = fabricated = 0
cf_total = cf_follow = 0
ooc_total = ooc_refused = 0
in_total = in_answered = 0

for p in probes:
    r = sear.ask(p["question"], narrator=False).report
    kept = [] if r.failed else r.verified_claims
    emitted += len(r.claims)
    fabricated += sum(not c.verified for c in r.claims)
    answer = " ".join(c.text for c in kept).lower()
    if p["layer"] == "out_of_corpus":
        ooc_total += 1
        ooc_refused += r.failed
    else:
        in_total += 1
        in_answered += not r.failed
    if p["layer"] == "counterfactual":
        cf_total += 1
        cf_follow += bool(p.get("expect")) and p["expect"].lower() in answer

pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "n/a"
print("pre-gate unverified claim rate:", pct(fabricated, emitted))
print("counterfactual adherence:      ", pct(cf_follow, cf_total))
print("failure-state recall (OOC):    ", pct(ooc_refused, ooc_total))
print("answer rate (in-corpus):       ", pct(in_answered, in_total))
