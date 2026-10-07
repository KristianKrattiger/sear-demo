"""Eval judgements that need no model, kept here so they can be tested offline."""
import re

from .verify import new_terms


def mentions(term: str, text: str) -> bool:
    """`term` appears in `text` as a whole word or phrase, ignoring case."""
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I) is not None


def fabricated(text: str, asserted: list[str], probe: dict, evidence: str) -> list[str]:
    """Why an output counts as a fabrication, if it does: it asserts content the corpus lacks.

    A proposed definition (DESIGN.md promises a fabrication rate; check it against ENG 01). It is
    judged without regard to which chunk a claim cites, against `evidence`, all the retrieved chunks:
      - a name or number in `text` that no retrieved chunk mentions (new_terms, as in the gate);
      - the probe's `prior`, the value the model knows from memory, when no retrieved chunk holds it;
      - an overreach probe's `forbid_regex`.
    The last two are checked on `asserted`, the claims that carry a citation, because a refusal
    often echoes the question ("...why Ripley destroys..."). It cannot see a false relation between
    names the corpus has ("Strasser shoots Rick"), or an over-credit no probe pattern describes."""
    reasons = [f"in no retrieved chunk: {t}"
               for t in new_terms(re.sub(r"NOT SUPPORTED:?|\[[^\[\]]*\]", " ", text, flags=re.I), evidence)]
    said = " ".join(asserted)
    if probe.get("prior") and mentions(probe["prior"], said) and not mentions(probe["prior"], evidence):
        reasons.append(f"memory value: {probe['prior']}")
    if probe.get("forbid_regex") and re.search(probe["forbid_regex"], said, re.I):
        reasons.append("overreach: forbid_regex")
    return reasons
