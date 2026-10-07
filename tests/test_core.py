from sear_demo.gate import gate
from sear_demo.schema import Claim
from sear_demo.segment import segment


def test_segment_extracts_citations():
    claims = segment("Alien was released in 1979 [f1-c1]. It was directed by Ridley Scott [f1-c2, f1-c3].")
    assert [c.cited for c in claims] == [["f1-c1"], ["f1-c2", "f1-c3"]]
    assert claims[0].text == "Alien was released in 1979."


def test_segment_keeps_initials_together():
    claims = segment('The creature was designed by H. R. Giger [alien-c03].')
    assert len(claims) == 1
    assert claims[0].text == "The creature was designed by H. R. Giger."
    assert claims[0].cited == ["alien-c03"]


def test_segment_uncited_tail_is_kept():
    claims = segment("Supported claim [a]. Unsupported tail.")
    assert [c.cited for c in claims] == [["a"], []]


def test_gate_fails_below_phi():
    claims = [Claim("a", ["x"], verified=True), Claim("b", ["y"]), Claim("c", ["z"])]
    assert gate(claims, phi=0.5) is False
    assert gate(claims, phi=0.3) is True


def test_gate_fails_on_empty():
    assert gate([], phi=0.5) is False


def _word_verifier():
    """A Verifier whose 'NLI' says entailed when every word of the claim appears in the premise."""
    import sys, types
    sys.modules.setdefault("transformers", types.SimpleNamespace(pipeline=None))
    from sear_demo.verify import Verifier

    v = Verifier.__new__(Verifier)
    v.tau = 0.5
    words = lambda t: set(t.lower().replace(".", "").split())
    v.entailment = lambda premise, hyp: float(words(hyp) <= words(premise))
    return v


def test_repair_recites_wrong_chunk():
    from sear_demo.schema import Chunk
    a, b = Chunk("a", "Ripley is the hero."), Chunk("b", "Holm plays Ash.")
    claims = _word_verifier().check([Claim("Holm plays Ash.", ["a"])], {"a": a, "b": b})
    assert not claims[0].verified
    _word_verifier().repair(claims, [a, b])
    assert claims[0].verified and claims[0].cited == ["b"] and claims[0].model_cited == ["a"]


def test_repair_finds_two_hop_pair_and_leaves_unsupported():
    from sear_demo.schema import Chunk
    a, b = Chunk("a", "Brody is the police chief."), Chunk("b", "Scheider plays Brody.")
    claims = [Claim("Scheider plays Brody the police chief.", ["b"]), Claim("Quint owns the Pequod.", ["a"])]
    v = _word_verifier()
    v.check(claims, {"a": a, "b": b})
    v.repair(claims, [a, b])
    assert claims[0].verified and set(claims[0].cited) == {"a", "b"}
    assert not claims[1].verified and claims[1].model_cited is None


def test_new_terms_finds_names_and_numbers_the_evidence_lacks():
    from sear_demo.verify import new_terms
    ev = ("This is about the film Alien. Alien is a film directed by Ridley Scott, released in 1983. "
          "Sigourney Weaver plays Ellen Ripley; H. R. Giger designed the creature.")
    assert new_terms("Alien was released in 1979.", ev) == ["1979"]
    assert new_terms("The film won the Academy Award for Best Animated Feature.",
                     "The film won the Academy Award for Best Picture.") == ["Best Animated Feature"]
    assert new_terms("Rick shoots the German Major Laszlo.", "Rick shoots the German Major Strasser.") == ["German Major Laszlo"]
    assert new_terms("The Nostromo crew meets Giger.", ev) == ["Nostromo"]  # "The" dropped, the name kept
    # a lone capitalised opener, possessives, initials and lowercase paraphrase pass
    assert new_terms("Directed by Ridley Scott, it stars Weaver's Ripley. Overall, Giger's design is iconic.", ev) == []
    assert new_terms("H. R. Giger designed it for director ridley scott.", ev) == []


def test_gate_withholds_a_swapped_name_that_nli_entails():
    """The real NLI read "Best Animated Feature" as entailed by "Best Picture" (0.991); repair finds no chunk naming it."""
    from sear_demo.schema import Chunk
    c = Chunk("spirit-c03", "The film won the Academy Award for Best Picture.", meta={"film": "Spirited Away"})
    v = _word_verifier()
    v.entailment = lambda premise, hyp: 0.99
    claims = v.check([Claim("Spirited Away won the Academy Award for Best Animated Feature.", ["spirit-c03"]),
                      Claim("Spirited Away won the Academy Award for Best Picture.", ["spirit-c03"])], {c.id: c})
    assert [(x.verified, x.new_terms) for x in claims] == [(False, ["Best Animated Feature"]), (True, [])]
    v.repair(claims, [c])
    assert not claims[0].verified and claims[0].model_cited is None


def test_repair_recites_a_name_to_the_chunk_that_has_it():
    from sear_demo.schema import Chunk
    a, b = Chunk("a", "The captain is Dallas."), Chunk("b", "Tom Skerritt plays the captain.")
    v = _word_verifier()
    v.entailment = lambda premise, hyp: 0.99  # an NLI that entails anything: only the names decide
    [claim] = v.check([Claim("Tom Skerritt plays the captain.", ["a"])], {"a": a, "b": b})
    assert claim.new_terms == ["Tom Skerritt"]
    v.repair([claim], [a, b])
    assert claim.verified and claim.cited == ["b"] and claim.new_terms == []


def test_focused_joint_premise_picks_matching_sentences():
    from sear_demo.schema import Chunk
    Verifier = type(_word_verifier())
    a = Chunk("a", "The crew fly the Nostromo. They eat dinner.")
    b = Chunk("b", "Weaver plays Ripley. Skerritt plays Dallas.")
    joint = Verifier.focused_joint_premise([a, b], "Skerritt plays Dallas of the Nostromo.")
    assert "Nostromo" in joint and "Skerritt plays Dallas" in joint and "dinner" not in joint


def _report(*claims):
    from sear_demo.schema import Chunk, Report
    retrieved = [Chunk("alien-c04", "Alien was released in 1983."), Chunk("alien-c03", "Giger designed the creature.")]
    return Report(query="Tell me about Alien.", retrieved=retrieved, claims=list(claims))


def test_format_report_shows_handles_for_verified_claims_only():
    from sear_demo.prompts import NARRATE, format_report
    r = _report(Claim("Alien was released in 1983.", ["alien-c04"], verified=True),
                Claim("Dropped claim.", ["alien-c03"]),
                Claim("Giger designed the creature.", ["alien-c03"], verified=True))
    assert format_report(r) == "[C1] Alien was released in 1983.\n[C2] Giger designed the creature."
    assert "alien-c0" not in format_report(r)  # the narrator never sees chunk IDs
    assert "{query}" in NARRATE and "{report}" in NARRATE


def test_split_sentences_keeps_initials_titles_and_closing_quotes():
    from sear_demo.segment import split_sentences
    assert split_sentences("The creature, by H. R. Giger, has a life cycle. Dr. Hooper agrees!") == [
        "The creature, by H. R. Giger, has a life cycle.", "Dr. Hooper agrees!"]
    assert split_sentences('Renault says "round up the usual suspects." Rick walks off. Was it 1942? 1943 it was.') == [
        'Renault says "round up the usual suspects."', "Rick walks off.", "Was it 1942?", "1943 it was."]
    assert split_sentences("One sentence only.") == ["One sentence only."]


def test_handles_are_one_per_sentence_with_the_claims_sources():
    r = _report(Claim("Alien was released in 1983. Giger designed the creature.", ["alien-c04", "alien-c03"],
                      verified=True, score=0.8),
                Claim("Dropped claim. Also dropped.", ["alien-c03"]))
    hs = r.handles
    assert list(hs) == ["C1", "C2"] and hs["C2"].text == "Giger designed the creature."
    assert all(c.cited == ["alien-c04", "alien-c03"] and c.verified and c.score == 0.8 for c in hs.values())
    hs["C1"].cited.append("x")
    assert r.claims[0].cited == ["alien-c04", "alien-c03"]  # handles never alias the claim's list
    [(_, _, chunks)] = r.chain(["C2"])
    assert [c.id for c in chunks] == ["alien-c04", "alien-c03"]


def test_chain_resolves_handles_to_claims_and_chunks():
    r = _report(Claim("Alien was released in 1983.", ["alien-c04"], verified=True))
    [(h, claim, chunks)] = r.chain(["C1", "C9"])
    assert h == "C1" and claim.cited == ["alien-c04"] and [c.id for c in chunks] == ["alien-c04"]


def test_repair_records_its_best_failed_candidate():
    """p15: repair tried the pair that names everything, and the verifier still scored it below tau."""
    from sear_demo.schema import Chunk
    a, b = Chunk("a", "Skerritt plays Captain Dallas."), Chunk("b", "The crew fly the Nostromo.")
    v = _word_verifier()
    v.entailment = lambda premise, hyp: 0.07 if "Skerritt" in premise and "Nostromo" in premise else 0.0
    [claim] = v.check([Claim("Skerritt plays Captain Dallas of the Nostromo.", ["a"])], {"a": a, "b": b})
    v.repair([claim], [a, b])
    assert not claim.verified and claim.model_cited is None
    assert claim.repair_best == {"cited": ["a", "b"], "score": 0.07, "new_terms": [], "tried": 3}
    fixed = v.check([Claim("Holm plays Ash.", ["a"])], {"a": a, "b": Chunk("b", "Holm plays Ash.")})
    _word_verifier().repair(fixed, [a, Chunk("b", "Holm plays Ash.")])
    assert fixed[0].verified and fixed[0].repair_best is None
