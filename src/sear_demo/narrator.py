"""The narrator: a plain-language summary of a gated report.

It sees the report only, never the corpus: the prompt holds the verified claims
under handles C1..Cn, and each summary sentence is checked against the claims
it cites, not against the chunks behind them.
"""
from .prompts import NARRATE, format_report
from .segment import segment
from .schema import Chunk, Claim, Narration, Report


def as_chunk(handle: str, claim: Claim, by_id: dict[str, Chunk]) -> Chunk:
    """A verified claim as a pseudo-chunk for the verifier, with the film of its first known source."""
    film = next((by_id[i].meta.get("film") for i in claim.cited if i in by_id), None)
    return Chunk(id=handle, text=claim.text, meta={"film": film} if film else {})


def narrate(report: Report, llm, verifier, by_id: dict[str, Chunk], min_claims: int = 2,
            contradict_tau: float = 0.5) -> Narration:
    """Summarise a gated report and label each sentence: supported, contradicts, unsupported or uncited.

    `llm` is any callable prompt -> text or (text, trace). `by_id` only lends
    the verifier each claim's film; the narrator never sees it. Citations are
    not repaired, so the narrator's slips stay visible.
    """
    if report.failed:
        return Narration([], skipped="the gate emitted the failure state")
    handles = report.handles
    if len(handles) < min_claims:
        n = len(handles)
        return Narration([], skipped=f"only {n} verified claim" + ("s" if n != 1 else ""))
    text = llm(NARRATE.format(query=report.query, report=format_report(report)))
    if isinstance(text, tuple):
        text = text[0]
    sentences = segment(text)
    unknown = sorted({h for s in sentences for h in s.cited if h not in handles})
    for s in sentences:
        s.cited = list(dict.fromkeys(h for h in s.cited if h in handles))  # valid, deduplicated
        if not s.cited:
            s.verified, s.score, s.label = False, 0.0, "uncited"
            continue
        sources = [as_chunk(h, handles[h], by_id) for h in s.cited]
        s.score = verifier.support(s.text, sources)
        # joint_premise of one chunk is the claim with the film prefix the verifier's premises use
        s.contradicts = [c.id for c in sources
                         if verifier.contradiction(verifier.joint_premise([c]), s.text) > contradict_tau]
        s.verified = s.score >= verifier.tau and not s.contradicts
        s.label = "contradicts" if s.contradicts else "supported" if s.verified else "unsupported"
    return Narration(sentences=sentences, unknown_ids=unknown)
