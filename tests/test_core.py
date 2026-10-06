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
