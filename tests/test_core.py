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
