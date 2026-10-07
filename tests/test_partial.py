"""Partial answers (generator.partial_answers): answer the supported part, decline the rest."""
from sear_demo.prompts import GENERATE, GENERATE_PARTIAL
from sear_demo.segment import answer_and_declines


def test_a_not_supported_line_is_a_decline_and_the_cited_part_is_the_answer():
    claims, declined = answer_and_declines(
        "Casablanca won the Academy Award for Best Picture [casa-c03].\nNOT SUPPORTED: the year it won.")
    assert [(c.text, c.cited) for c in claims] == [("Casablanca won the Academy Award for Best Picture.", ["casa-c03"])]
    assert declined == ["the year it won"]


def test_uncited_sentences_about_what_the_sources_lack_are_declines_too():
    claims, declined = answer_and_declines(
        "Herrmann composed the two-note theme [jaws-c03]. The sources do not say who composed the full score. NOT SUPPORTED.")
    assert [c.cited for c in claims] == [["jaws-c03"]]
    assert declined == ["", "The sources do not say who composed the full score."]
    assert answer_and_declines("The sources don't name the actor.")[1] == ["The sources don't name the actor."]


def test_a_bare_refusal_leaves_no_claims():
    assert answer_and_declines("NOT SUPPORTED") == ([], [""])
    assert answer_and_declines("not supported.")[0] == []


def test_other_uncited_text_stays_a_claim_that_cannot_verify():
    """An uncited answer is not a decline: it reaches the gate, which drops it (a gate withhold, not a refusal)."""
    claims, declined = answer_and_declines("Victor Laszlo is played by Paul Henreid.\nNOT SUPPORTED: anything else.")
    assert [(c.text, c.cited) for c in claims] == [("Victor Laszlo is played by Paul Henreid.", [])]
    assert declined == ["anything else"]


def test_the_partial_prompt_keeps_the_strict_prompts_rules():
    for rule in ("using ONLY the sources", "Every sentence", "cite every source", "reply exactly: NOT SUPPORTED"):
        assert rule in GENERATE and rule in GENERATE_PARTIAL
    assert "{sources}" in GENERATE_PARTIAL and "{query}" in GENERATE_PARTIAL and "NOT SUPPORTED: <" in GENERATE_PARTIAL
