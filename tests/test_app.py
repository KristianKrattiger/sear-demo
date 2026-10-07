"""The Streamlit app, headless, with a Sear that loads no model."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from sear_demo import pipeline  # noqa: E402
from sear_demo.schema import Chunk, Claim, Narration, Report  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


class FakeSear:
    """Answers from whichever corpus is in use: Druillet on the poisoned fork, Giger elsewhere."""

    def __init__(self, cfg):
        self.cfg, self.chunks_path, self.switched, self.answered = cfg, Path(cfg["corpus"]["chunks"]), [], []

    def use_corpus(self, folder):
        if Path(folder) / "chunks.jsonl" != self.chunks_path:
            self.chunks_path = Path(folder) / "chunks.jsonl"
            self.switched.append(Path(folder).name)

    def answer(self, query):
        corpus = self.chunks_path.parent.name
        self.answered.append(corpus)
        who = "Druillet" if corpus == "poisoned" else "Giger"
        chunk = Chunk("alien-c03", f"The creature was designed by {who}.", meta={"film": "Alien"})
        return Report(query, [chunk], [Claim(f"{who} designed the creature.", ["alien-c03"], verified=True, score=0.99)])

    def narrate(self, report):
        return Narration([], skipped="only 1 verified sentence")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(pipeline, "Sear", FakeSear)
    import streamlit as st
    st.cache_resource.clear()
    at = st_testing.AppTest.from_file(APP, default_timeout=30)
    return at.run()


def texts(at):
    return " ".join(m.value for m in at.markdown)


def test_the_corpus_selector_swaps_corpora_without_reloading_the_app(app):
    assert not app.exception
    picker = app.segmented_control(key="corpus")
    assert picker.value == "corpus" and set(picker.options) >= {"main", "poisoned", "conflict"}
    app.text_input[0].set_value("Who designed the creature in Alien?").run()
    assert "Giger designed the creature." in texts(app)
    picker.set_value("poisoned").run()
    assert "Druillet designed the creature." in texts(app) and "Giger" not in texts(app)
    assert not app.exception


def test_reports_are_cached_per_corpus_and_question(app):
    app.text_input[0].set_value("Who designed the creature in Alien?").run()
    app.segmented_control(key="corpus").set_value("poisoned").run()
    app.segmented_control(key="corpus").set_value("corpus").run()
    assert "Giger designed the creature." in texts(app)
    assert app.session_state["reports"].keys() == {("corpus", "Who designed the creature in Alien?"),
                                                   ("poisoned", "Who designed the creature in Alien?")}


def test_compare_view_shows_both_corpora_side_by_side(app):
    app.toggle[0].set_value(True).run()
    app.text_input[0].set_value("Who designed the creature in Alien?").run()
    assert not app.exception
    assert [h.value for h in app.subheader] == ["main", "poisoned"]
    assert "Giger designed the creature." in texts(app) and "Druillet designed the creature." in texts(app)
    assert "(entailment 0.99)" in texts(app)


def test_a_citation_opens_its_chunk(app):
    """DESIGN.md's demo step 1: each cited chunk ID is a popover holding the chunk."""
    app.text_input[0].set_value("Who designed the creature in Alien?").run()
    [popover] = app.get("popover")
    assert [m.value for m in popover.markdown] == ["The creature was designed by Giger."]
