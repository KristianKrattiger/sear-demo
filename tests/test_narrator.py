import re

from sear_demo.narrator import as_chunk, narrate
from sear_demo.schema import Chunk, Claim, Narration, Report
from sear_demo.verify import Verifier

BY_ID = {"alien-c04": Chunk("alien-c04", "Alien was released in 1983.", meta={"film": "Alien"}),
         "alien-c03": Chunk("alien-c03", "Giger designed the creature.", meta={"film": "Alien"})}


def _report(n=2, failed=False):
    claims = [Claim("Alien was released in 1983.", ["alien-c04"], verified=True),
              Claim("Giger designed the creature.", ["alien-c03"], verified=True)]
    return Report(query="Tell me about Alien.", retrieved=list(BY_ID.values()), claims=claims[:n], failed=failed)


def _verifier():
    """Word-subset 'NLI' as in test_core, plus: changing the premise's year is a contradiction."""
    v = Verifier.__new__(Verifier)
    v.tau = 0.5
    words = lambda t: set(t.lower().replace(".", "").split())
    years = lambda t: set(re.findall(r"\b\d{4}\b", t))
    v.entailment = lambda premise, hyp: float(words(hyp) <= words(premise))
    v.contradiction = lambda premise, hyp: float(bool(years(hyp) - years(premise)) and bool(years(premise) - years(hyp)))
    return v


class FakeLLM:
    def __init__(self, out):
        self.out, self.prompts = out, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.out


def _narrate(text, **kw):
    return narrate(_report(), FakeLLM(text), _verifier(), BY_ID, **kw)


def test_restated_claims_are_supported():
    llm = FakeLLM("Alien was released in 1983 [C1]. Giger designed the creature [C2].")
    n = narrate(_report(), llm, _verifier(), BY_ID)
    assert [(s.label, s.cited, s.verified) for s in n.sentences] == [("supported", ["C1"], True), ("supported", ["C2"], True)]
    assert n.unknown_ids == [] and n.skipped == ""
    [prompt] = llm.prompts
    assert "Tell me about Alien." in prompt and "[C1] Alien was released in 1983." in prompt and "alien-c0" not in prompt


def test_changed_year_contradicts():
    [s] = _narrate("Alien was released in 1979 [C1].").sentences
    assert s.label == "contradicts" and s.contradicts == ["C1"] and not s.verified


def test_contradiction_outweighs_entailment():
    v = _verifier()
    v.entailment = lambda premise, hyp: 1.0
    [s] = narrate(_report(), FakeLLM("Alien was released in 1979 [C1]."), v, BY_ID).sentences
    assert s.score == 1.0 and s.label == "contradicts" and not s.verified


def test_added_fact_is_unsupported():
    [s] = _narrate("Giger designed the creature in Switzerland [C2].").sentences
    assert s.label == "unsupported" and s.contradicts == [] and not s.verified


def test_chunk_ids_and_unknown_handles_are_uncited():
    n = _narrate("Giger designed the creature [alien-c03]. Alien was released in 1983 [C9]. "
                 "Alien was released in 1983 [C1][C9].")
    assert [(s.label, s.cited, s.score, s.verified) for s in n.sentences] == [
        ("uncited", [], 0.0, False), ("uncited", [], 0.0, False), ("supported", ["C1"], 1.0, True)]
    assert n.unknown_ids == ["C9", "alien-c03"]


def test_failed_and_thin_reports_skip_without_calling_the_llm():
    llm = FakeLLM("Alien was released in 1983 [C1].")
    assert narrate(_report(failed=True), llm, _verifier(), BY_ID) == Narration([], skipped="the gate emitted the failure state")
    assert narrate(_report(n=1), llm, _verifier(), BY_ID) == Narration([], skipped="only 1 verified claim")
    assert narrate(_report(), llm, _verifier(), BY_ID, min_claims=3).skipped == "only 2 verified claims"
    assert llm.prompts == []


def test_llm_may_return_text_and_trace():
    [s] = _narrate(("Alien was released in 1983 [C1].", {"attentions": None})).sentences
    assert s.label == "supported"


def test_pseudo_chunk_carries_the_film_into_the_premise():
    claim = Claim("Alien was released in 1983.", ["gone", "alien-c04"], verified=True)
    assert as_chunk("C1", claim, BY_ID) == Chunk("C1", "Alien was released in 1983.", meta={"film": "Alien"})
    assert as_chunk("C1", Claim("x.", ["gone"]), BY_ID).meta == {}
    v, seen = _verifier(), []
    contradiction = v.contradiction
    v.contradiction = lambda premise, hyp: seen.append(premise) or contradiction(premise, hyp)
    narrate(_report(), FakeLLM("Alien was released in 1979 [C1]."), v, BY_ID)
    assert seen == ["This is about the film Alien. Alien was released in 1983."]


def test_verifier_reads_entailment_and_contradiction_from_one_nli_call():
    v, calls = Verifier.__new__(Verifier), []
    v.nli = lambda pair: calls.append(pair) or [{"label": "CONTRADICTION", "score": 0.7},
                                                {"label": "NEUTRAL", "score": 0.2}, {"label": "ENTAILMENT", "score": 0.1}]
    assert v.scores("p", "h") == {"contradiction": 0.7, "neutral": 0.2, "entailment": 0.1}
    assert (v.entailment("p", "h"), v.contradiction("p", "h")) == (0.1, 0.7)
    assert calls == [{"text": "p", "text_pair": "h"}] * 3


def test_repeated_handle_is_cited_once():
    n = _narrate("Alien was released in 1983 [C1][C1].")
    assert n.sentences[0].cited == ["C1"] and n.sentences[0].label == "supported"
