import re

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


def test_conflict_outcome_says_whether_both_values_reach_the_reader():
    from sear_demo.metrics import conflict_outcome
    from sear_demo.schema import Chunk, Claim, Report
    probe = {"id": "k01", "values": {"Orca": r"\borca\b", "Pequod": r"\bpequod\b"}, "chunks": ["jaws-c02", "jaws-c04"],
             "prior": "Orca"}
    retrieved = [Chunk("jaws-c02", "Quint's boat, the Orca."), Chunk("jaws-c01", "Amity Island.")]
    picked = Report("q", retrieved, [Claim("Quint's boat is the Orca.", ["jaws-c02"], verified=True),
                                     Claim("Some call it the Pequod.", ["jaws-c02"])], raw="... Orca ... Pequod ...")
    row = conflict_outcome(probe, picked, "Quint's boat is the Orca.")
    assert row["outcome"] == "picked Orca" and row["raw"] == ["Orca", "Pequod"] and row["retrieved"] == ["jaws-c02"]
    assert row["cited"] == ["jaws-c02"] and row["dropped"] == ["Some call it the Pequod."]
    both = conflict_outcome(probe, picked, "The Orca [jaws-c02], or the Pequod [jaws-c04].")
    assert both["outcome"] == "surfaced both"
    assert conflict_outcome(probe, Report("q", retrieved, [], failed=True), "")["outcome"] == "refused"


def test_the_conflict_fork_is_the_main_corpus_plus_conflicting_chunks():
    from pathlib import Path
    from sear_demo.corpus import load_chunks, load_probes
    root = Path(__file__).resolve().parents[1] / "corpus"
    main, fork = load_chunks(root / "chunks.jsonl"), load_chunks(root / "conflict" / "chunks.jsonl")
    assert fork[:len(main)] == main  # nothing in the main corpus is changed
    ids = {c.id for c in fork}
    by_id = {c.id: c for c in fork}
    for p in load_probes(root / "conflict" / "probes.jsonl"):
        assert p["layer"] == "conflict" and len(p["values"]) == 2 and set(p["chunks"]) <= ids
        # each value's pattern finds it in the chunk that states it, and not in the other one
        (_, mine), (_, theirs) = p["values"].items()
        a, b = (by_id[i].text for i in p["chunks"])
        assert re.search(mine, a, re.I) and re.search(theirs, b, re.I)
        assert not re.search(mine, b, re.I) and not re.search(theirs, a, re.I)


def test_a_name_the_question_gives_is_not_a_fabrication():
    """6fdf05d-partial: 'NOT SUPPORTED: Who directed the sequel "Aliens."' was flagged for echoing the question."""
    refusal = 'The sources do not say who directed the sequel "Aliens." NOT SUPPORTED: Who directed the sequel "Aliens."'
    corpus_only = "This is about the film Alien. Alien is a film directed by Ridley Scott."
    assert fabricated(refusal, [], {}, corpus_only) == ["in no retrieved chunk: Aliens"]
    assert fabricated(refusal, [], {}, corpus_only + " Who directed the sequel Aliens?") == []
