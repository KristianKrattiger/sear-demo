from .prompts import NARRATE, format_report
from .segment import segment
from .schema import Chunk, Narration, Report


def narrate(report: Report, llm, verifier, by_id: dict[str, Chunk]) -> Narration | None:
    """Summarise a gated report. Sees the report only, never the corpus.

    `llm` is any callable prompt -> text (local Generator or an API wrapper).
    """
    if report.failed or not report.verified_claims:
        return None
    text = llm(NARRATE.format(report=format_report(report)))
    if isinstance(text, tuple):
        text = text[0]
    sentences = segment(text)
    allowed = report.cited_ids
    unknown = sorted({i for s in sentences for i in s.cited if i not in allowed})
    for s in sentences:
        s.cited = [i for i in s.cited if i in allowed]
    # Re-verify each summary sentence against the chunks the report cites.
    verifier.check(sentences, {i: by_id[i] for i in allowed if i in by_id})
    return Narration(sentences=sentences, unknown_ids=unknown)
