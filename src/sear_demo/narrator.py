"""The narrator: a plain-language summary of a gated report.

It sees the report only, never the corpus: the prompt holds the verified claims
under handles C1..Cn (one per sentence), and each summary sentence is checked against the claims
it cites, not against the chunks behind them.
"""
import re
from itertools import combinations

from .prompts import NARRATE, format_report
from .segment import segment
from .schema import Chunk, Claim, Narration, Report
from .verify import new_terms


def as_chunk(handle: str, claim: Claim, by_id: dict[str, Chunk]) -> Chunk:
    """A verified claim as a pseudo-chunk for the verifier, with the film of its first known source."""
    film = next((by_id[i].meta.get("film") for i in claim.cited if i in by_id), None)
    return Chunk(id=handle, text=claim.text, meta={"film": film} if film else {})


def backing(handle: Claim, by_id: dict[str, Chunk], verifier) -> list[str]:
    """The fewest chunks that back this sentence, so its source chain doesn't list every chunk its
    paragraph cited: those that entail it alone, else the first pair that entails it together, else all."""
    return backing_detail(handle, by_id, verifier)[0]


def backing_detail(handle: Claim, by_id: dict[str, Chunk], verifier) -> tuple[list[str], bool]:
    """backing(), and whether those chunks were shown to entail the sentence.

    False means no cited chunk alone and no pair of them entails it: the sentence was verified only
    as part of its paragraph, and the chain falls back to every chunk the paragraph cited. The UI and
    the log mark those handles instead of implying the listed chunks back the sentence."""
    known = [i for i in handle.cited if i in by_id]
    own = [i for i in known
           if any(verifier.entailment(p, handle.text) >= verifier.tau for p in verifier.premises(by_id[i]))]
    if own:
        return own, True
    if len(known) > 2:
        for pair in combinations(known, 2):
            if verifier.support(handle.text, [by_id[i] for i in pair]) >= verifier.tau:
                return list(pair), True
    elif len(known) == 2:  # the pair is all of them, so the chain is the same either way
        return list(handle.cited), verifier.support(handle.text, [by_id[i] for i in known]) >= verifier.tau
    return list(handle.cited), False


def contradicted(text: str, sources: list[Chunk], verifier, tau: float) -> list[str]:
    """Handles the sentence contradicts, judged against all its cited claims together.

    A sentence that merges C2 and C3 says things C2 alone doesn't, and NLI on that partial
    premise can read the extra as conflict. So the cited claims together must contradict it;
    then the badge names the ones that also do alone, or all of them if none does.
    """
    # joint_premise of one chunk is the claim with the film prefix the verifier's premises use
    if verifier.contradiction(verifier.joint_premise(sources), text) <= tau:
        return []
    if len(sources) == 1:
        return [sources[0].id]
    alone = [c.id for c in sources if verifier.contradiction(verifier.joint_premise([c]), text) > tau]
    return alone or [c.id for c in sources]


def label(s: Claim, handles: dict[str, Claim], by_id: dict[str, Chunk], verifier, contradict_tau: float = 0.5) -> Claim:
    """Check one summary sentence against the handles it cites and label it in place.

    Supported needs entailment, no contradiction, and no name or number the cited handles lack."""
    s.cited = list(dict.fromkeys(h for h in s.cited if h in handles))  # valid, deduplicated
    if not s.cited:
        s.verified, s.score, s.label = False, 0.0, "uncited"
        return s
    sources = [as_chunk(h, handles[h], by_id) for h in s.cited]
    s.score = verifier.support(s.text, sources)
    s.contradicts = contradicted(s.text, sources, verifier, contradict_tau)
    # the film prefix counts as evidence: "Spirited Away won..." may cite a handle that says "The film won..."
    s.new_terms = new_terms(s.text, verifier.joint_premise(sources))
    s.verified = s.score >= verifier.tau and not s.contradicts and not s.new_terms
    s.label = "contradicts" if s.contradicts else "supported" if s.verified else "unsupported"
    return s


def seeded(report: Report, find: str, replace: str) -> Claim | None:
    """A control slip: the first verified sentence holding `find`, with it swapped for `replace`,
    citing that sentence's handle. Labelling it shows whether the badge fires on a known slip."""
    for h, c in report.handles.items():
        if find.lower() in c.text.lower():
            return Claim(re.sub(re.escape(find), replace, c.text, count=1, flags=re.I), [h])
    return None


def control(report: Report, find: str, replace: str, also: list[str] = ()) -> Claim | None:
    """seeded(), also citing the first handle holding each string in `also`, so a control can swap in
    a name from another cited claim. None when any of them is missing: the control doesn't apply."""
    s = seeded(report, find, replace)
    for a in also if s else ():
        h = next((h for h, c in report.handles.items() if a.lower() in c.text.lower()), None)
        if h is None:
            return None
        s.cited += [h] if h not in s.cited else []
    return s


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
        return Narration([], skipped=f"only {n} verified sentence" + ("s" if n != 1 else ""))
    text = llm(NARRATE.format(query=report.query, report=format_report(report)))
    if isinstance(text, tuple):
        text = text[0]
    sentences = segment(text)
    unknown = sorted({h for s in sentences for h in s.cited if h not in handles})
    for s in sentences:
        label(s, handles, by_id, verifier, contradict_tau)
    used = dict.fromkeys(h for s in sentences for h in s.cited)
    backed = {h: backing_detail(handles[h], by_id, verifier) for h in used}
    return Narration(sentences=sentences, unknown_ids=unknown, sources={h: ids for h, (ids, _) in backed.items()},
                     unbacked=[h for h, (_, ok) in backed.items() if not ok])
