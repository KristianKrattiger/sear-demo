from sear_demo.metrics import fabricated, mentions

EVIDENCE = ("This is about the film Casablanca. Casablanca is a romantic drama directed by Michael Curtiz. "
            "His former lover Ilsa Lund arrives with her husband Victor Laszlo, a resistance leader.")


def test_a_name_no_retrieved_chunk_mentions_is_a_fabrication():
    """p20 on the ffaaad4 run: the generator named an actor the corpus never mentions."""
    raw = "Victor Laszlo in Casablanca is played by Paul Henreid [casa-c01]."
    assert fabricated(raw, ["Victor Laszlo in Casablanca is played by Paul Henreid."], {}, EVIDENCE) == [
        "in no retrieved chunk: Paul Henreid"]


def test_refusals_and_citations_are_not_fabrications():
    refusal = "The sources provided do not specify who plays Victor Laszlo in Casablanca. NOT SUPPORTED."
    assert fabricated(refusal, [], {"forbid_regex": "laszlo"}, EVIDENCE) == []  # an echo of the question is not a claim
    assert fabricated("Ilsa Lund arrives with Victor Laszlo [casa-c01][casa-c02].", ["Ilsa Lund arrives with Victor Laszlo."],
                      {}, EVIDENCE) == []


def test_the_memory_value_counts_only_when_the_corpus_lacks_it():
    probe = {"prior": "1979"}
    assert fabricated("Alien was released in 1979.", ["Alien was released in 1979."], probe,
                      "This is about the film Alien. Alien was released in 1983.") == [
        "in no retrieved chunk: 1979", "memory value: 1979"]
    rick = {"prior": "Rick"}  # the poisoned fork's x04: the real value is also a name the corpus uses
    assert fabricated("Ilsa shoots Strasser while Rick watches.", ["Ilsa shoots Strasser while Rick watches."], rick,
                      "Rick gives up Ilsa. Ilsa shoots the German Major Strasser.") == []


def test_an_overreach_pattern_in_a_cited_claim_is_a_fabrication():
    probe = {"forbid_regex": r"herrmann[^.]*\b(score|music)\b|\b(score|music)\b[^.]*herrmann"}
    claim = "The full orchestral score of Jaws was composed by Bernard Herrmann."
    assert fabricated(claim, [claim], probe, "This is about the film Jaws. The two-note theme was composed by Bernard Herrmann.") == [
        "overreach: forbid_regex"]


def test_mentions_matches_whole_words_only():
    assert mentions("Sen", "calls her Sen.") and not mentions("Sen", "its presence is suggested")
