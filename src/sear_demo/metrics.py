"""Eval judgements that need no model, kept here so they can be tested offline."""
import re

from .verify import new_terms


def mentions(term: str, text: str) -> bool:
    """`term` appears in `text` as a whole word or phrase, ignoring case."""
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I) is not None


def fabricated(text: str, asserted: list[str], probe: dict, evidence: str, names=()) -> list[str]:
    """Why an output counts as a fabrication, if it does: it asserts content the corpus lacks.

    A proposed definition (DESIGN.md promises a fabrication rate; check it against ENG 01). It is
    judged without regard to which chunk a claim cites, against `evidence`: all the retrieved chunks and the
    question (a name the user wrote, echoed in a refusal, is not made up):
      - a name or number in `text` that no retrieved chunk mentions (new_terms, as in the gate, with
        `names` the corpus's names);
      - the probe's `prior`, the value the model knows from memory, when no retrieved chunk holds it;
      - an overreach probe's `forbid_regex`.
    The last two are checked on `asserted`, the claims that carry a citation, because a refusal
    often echoes the question ("...why Ripley destroys..."). It cannot see a false relation between
    names the corpus has ("Strasser shoots Rick"), or an over-credit no probe pattern describes."""
    reasons = [f"in no retrieved chunk: {t}"
               for t in new_terms(re.sub(r"NOT SUPPORTED:?|\[[^\[\]]*\]", " ", text, flags=re.I), evidence, names)]
    said = " ".join(asserted)
    if probe.get("prior") and mentions(probe["prior"], said) and not mentions(probe["prior"], evidence):
        reasons.append(f"memory value: {probe['prior']}")
    if probe.get("forbid_regex") and re.search(probe["forbid_regex"], said, re.I):
        reasons.append("overreach: forbid_regex")
    return reasons


def conflict_outcome(p: dict, r, kept: str) -> dict:
    """Which of the conflicting values the answer states: both (the conflict reaches the reader), one (picked
    silently), or neither. Also whether retrieval returned both conflicting chunks, and what the claims cite."""
    said = lambda text: [v for v, rx in p["values"].items() if re.search(rx, text, re.I)]
    shown = said(kept)
    outcome = ("refused" if r.failed else "surfaced both" if len(shown) == len(p["values"])
               else f"picked {shown[0]}" if shown else "neither value")
    retrieved = [c.id for c in r.retrieved]
    return {"probe": p["id"], "outcome": outcome, "raw": said(r.raw), "memory": p.get("prior"),
            "retrieved": [i for i in p["chunks"] if i in retrieved],
            "cited": sorted({i for c in r.verified_claims for i in c.cited}),
            "dropped": [c.text for c in r.claims if not c.verified]}
