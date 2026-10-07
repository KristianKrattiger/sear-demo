"""Sentence-level verification (verifier.granularity: sentence) and its antecedent remedies."""
import pytest

from sear_demo.schema import Chunk, Claim
from sear_demo.segment import first_name, resolve_reference, sentence_claims

FILMS = {"alien-c04": "Alien", "casa-c02": "Casablanca"}
PARAGRAPH = Claim("Alien is a horror film. It was released in 1983.", ["alien-c04"])


def test_each_sentence_becomes_a_claim_with_the_paragraphs_citations():
    claims = sentence_claims([PARAGRAPH, Claim("Uncited tail.", [])])
    assert [(c.text, c.cited, c.checked) for c in claims] == [
        ("Alien is a horror film.", ["alien-c04"], ""), ("It was released in 1983.", ["alien-c04"], ""),
        ("Uncited tail.", [], "")]
    claims[0].cited.append("x")
    assert claims[1].cited == ["alien-c04"] and PARAGRAPH.cited == ["alien-c04"]  # no shared lists


@pytest.mark.parametrize("context, checked", [
    ("none", ""),
    ("film", "This is about the film Alien. It was released in 1983."),
    ("previous", "Alien is a horror film. It was released in 1983."),
    ("resolve", "Alien was released in 1983."),
])
def test_context_sets_what_the_verifier_scores_and_never_the_shown_sentence(context, checked):
    second = sentence_claims([PARAGRAPH], context, FILMS)[1]
    assert second.text == "It was released in 1983." and second.checked == checked


def test_unknown_context_is_an_error():
    with pytest.raises(ValueError):
        sentence_claims([PARAGRAPH], "guess", FILMS)


def test_resolve_reference_uses_the_film_or_the_previous_sentences_first_name():
    assert resolve_reference("Its tagline is famous.", "", "Alien") == "Alien's tagline is famous."
    assert resolve_reference("He walks off with Renault.", "Rick shoots the German Major Strasser.", "Casablanca") == \
        "Rick walks off with Renault."
    assert resolve_reference("Her parents become pigs.", "The story follows Chihiro Ogino.", None) == \
        "Chihiro Ogino's parents become pigs."
    assert resolve_reference("They land.", "The crew wake up.", "Alien") == "They land."  # no name to point at
    assert resolve_reference("Ripley survives.", "Anything.", "Alien") == "Ripley survives."
    assert first_name("The Nostromo's crew wake.") == "Nostromo"


def _verifier(supported: set[str]):
    """The word-subset 'NLI' of the other tests, over the text the verifier is given."""
    import sys
    import types
    sys.modules.setdefault("transformers", types.SimpleNamespace(pipeline=None))
    from sear_demo.verify import Verifier
    v = Verifier.__new__(Verifier)
    v.tau = 0.5
    words = lambda t: set(t.lower().replace(".", "").split())
    v.entailment = lambda premise, hyp: float(words(hyp) <= words(premise))
    return v


def test_the_verifier_scores_checked_text_when_there_is_one():
    c = Chunk("alien-c04", "Alien was released in 1983.", meta={"film": "Alien"})
    bare, resolved = sentence_claims([PARAGRAPH], "none", FILMS)[1], sentence_claims([PARAGRAPH], "resolve", FILMS)[1]
    v = _verifier(set())
    v.check([bare, resolved], {c.id: c})
    assert not bare.verified  # "it" is not in the premise
    assert resolved.verified and resolved.text == "It was released in 1983."
