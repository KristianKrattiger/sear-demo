"""Sear.answer / narrate / ask without loading any model."""
import importlib
import sys
import types
from pathlib import Path

import pytest

from sear_demo.config import load_config
from sear_demo.schema import Chunk, Narration

CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
CHUNKS = [Chunk("alien-c01", "Alien was released in 1979."), Chunk("alien-c03", "Giger designed the creature.")]
ANSWER = "Alien was released in 1979 [alien-c01]. Giger designed the creature [alien-c03]."


def _stub_if_missing(name, **attrs):
    """Stand in for a module whose GPU-stack imports (torch, faiss, ...) are not installed."""
    try:
        importlib.import_module(name)
    except ImportError:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod


_stub_if_missing("sear_demo.generate", Generator=None)
_stub_if_missing("sear_demo.retrieve", Retriever=None)

from sear_demo import pipeline  # noqa: E402
from sear_demo.pipeline import Sear  # noqa: E402


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, query, k):
        self.calls += 1
        return CHUNKS[:k]


class FakeGenerator:
    def __init__(self, text=ANSWER):
        self.text, self.calls = text, 0

    def __call__(self, prompt):
        self.calls += 1
        return self.text, {"attentions": None, "prompt_len": 0}


class FakeVerifier:
    """Verified when the claim cites a chunk it was allowed to cite."""

    def check(self, claims, by_id):
        for c in claims:
            c.verified = any(i in by_id for i in c.cited)
            c.score = float(c.verified)
        return claims

    def repair(self, claims, retrieved):
        return claims


def _sear(enabled=False, text=ANSWER):
    cfg = load_config(CONFIG)
    cfg["narrator"].update(enabled=enabled, min_claims=3, contradict_tau=0.7)
    s = Sear.__new__(Sear)
    s.cfg, s.chunks, s.by_id = cfg, CHUNKS, {c.id: c for c in CHUNKS}
    s.retriever, s.generator, s.verifier = FakeRetriever(), FakeGenerator(text), FakeVerifier()
    s.narrator_llm = s.generator
    return s


@pytest.fixture
def calls(monkeypatch):
    calls = []

    def fake_narrate(report, llm, verifier, by_id, **kw):
        calls.append((report, llm, verifier, by_id, kw))
        return Narration(sentences=[])

    monkeypatch.setattr(pipeline, "narrate", fake_narrate)
    return calls


def test_ask_without_narrator_never_narrates(calls):
    result = _sear(enabled=True).ask("When was Alien released?", narrator=False)
    assert result.narration is None and calls == []
    assert [c.verified for c in result.report.claims] == [True, True] and not result.report.failed


def test_ask_with_narrator_narrates_the_answered_report(calls):
    s = _sear(enabled=False)
    result = s.ask("When was Alien released?", narrator=True)
    [(report, llm, verifier, by_id, kw)] = calls
    assert report is result.report and report == s.answer("When was Alien released?")
    assert llm is s.narrator_llm and verifier is s.verifier and by_id is s.by_id
    assert kw == {"min_claims": 3, "contradict_tau": 0.7}
    assert isinstance(result.narration, Narration)


@pytest.mark.parametrize("enabled", [True, False])
def test_ask_default_follows_config(calls, enabled):
    result = _sear(enabled=enabled).ask("When was Alien released?")
    assert len(calls) == int(enabled) and (result.narration is not None) == enabled


def test_failed_report_is_still_passed_to_narrator(calls):
    result = _sear(text="NOT SUPPORTED").ask("Who won the 2030 Oscar?", narrator=True)
    assert result.report.failed and result.report.claims == []
    assert result.report.failure_message == load_config(CONFIG)["gate"]["failure_message"]
    assert len(calls) == 1 and result.narration is not None  # narrate() reports why it skipped


def test_narrate_alone_does_not_regenerate(calls):
    s = _sear()
    report = s.ask("When was Alien released?", narrator=False).report
    assert (s.retriever.calls, s.generator.calls) == (1, 1)
    narration = s.narrate(report)
    assert (s.retriever.calls, s.generator.calls) == (1, 1)
    assert calls[0][0] is report and isinstance(narration, Narration)


def test_narrator_backend_checked_at_init(monkeypatch):
    monkeypatch.setattr(pipeline, "load_chunks", lambda path: list(CHUNKS))
    monkeypatch.setattr(pipeline, "Retriever", lambda *a, **k: FakeRetriever())
    monkeypatch.setattr(pipeline, "Generator", lambda *a, **k: FakeGenerator())
    monkeypatch.setattr(pipeline, "Verifier", lambda *a, **k: FakeVerifier())
    cfg = load_config(CONFIG)
    s = Sear(cfg)
    assert s.narrator_llm is s.generator
    del cfg["narrator"]["backend"]
    s = Sear(cfg)  # missing backend means local
    assert s.narrator_llm is s.generator
    cfg["narrator"]["backend"] = "api"
    with pytest.raises(NotImplementedError, match="'api' is not wired yet"):
        Sear(cfg)


def test_config_narrator_block():
    assert load_config(CONFIG)["narrator"] == {"enabled": False, "backend": "local", "min_claims": 2, "contradict_tau": 0.5}


class TextVerifier(FakeVerifier):
    """Verified unless the text it is given mentions Paris."""

    def check(self, claims, by_id):
        for c in claims:
            c.verified = "Paris" not in (c.checked or c.text)
            c.score = float(c.verified)
        return claims


def test_sentence_mode_verifies_each_sentence_and_phi_counts_sentences(calls):
    s = _sear(text="Alien was released in 1979. It was shot in Paris. Giger designed the creature [alien-c01].")
    s.verifier = TextVerifier()
    paragraph = s.answer("Tell me about Alien.")
    assert len(paragraph.claims) == 1 and paragraph.failed  # the default: one claim, dropped as a whole
    s.cfg["verifier"].update(granularity="sentence", sentence_context="none")
    sentences = s.answer("Tell me about Alien.")
    assert [c.verified for c in sentences.claims] == [True, False, True] and not sentences.failed  # 2 of 3 >= phi
    assert all(c.cited == ["alien-c01"] for c in sentences.claims)
    assert [h.text for h in sentences.handles.values()] == ["Alien was released in 1979.", "Giger designed the creature."]


def test_granularity_is_checked_at_init(monkeypatch):
    monkeypatch.setattr(pipeline, "load_chunks", lambda path: list(CHUNKS))
    monkeypatch.setattr(pipeline, "Retriever", lambda *a, **k: FakeRetriever())
    monkeypatch.setattr(pipeline, "Generator", lambda *a, **k: FakeGenerator())
    monkeypatch.setattr(pipeline, "Verifier", lambda *a, **k: FakeVerifier())
    cfg = load_config(CONFIG)
    assert cfg["verifier"]["granularity"] == "claim" and cfg["verifier"]["sentence_context"] == "none"  # off by default
    cfg["verifier"]["granularity"] = "word"
    with pytest.raises(ValueError, match="granularity"):
        Sear(cfg)


def test_partial_answers_gate_the_answered_part_and_keep_the_declines(calls):
    text = "Giger designed the creature [alien-c03].\nNOT SUPPORTED: who built the suit."
    strict = _sear(text=text).answer("Who designed and built the creature?")
    assert strict.failed and strict.claims == [] and strict.declined == []  # the default: NOT SUPPORTED fails it all
    s = _sear(text=text)
    s.cfg["generator"]["partial_answers"] = True
    report = s.answer("Who designed and built the creature?")
    assert not report.failed and [c.text for c in report.verified_claims] == ["Giger designed the creature."]
    assert report.declined == ["who built the suit"]


def test_partial_answers_still_fail_when_nothing_is_answered(calls):
    s = _sear(text="The sources do not say who plays Laszlo. NOT SUPPORTED.")
    s.cfg["generator"]["partial_answers"] = True
    report = s.answer("Who plays Victor Laszlo?")
    assert report.failed and report.claims == [] and len(report.declined) == 2
    s.generator.text = "Paul Henreid plays Victor Laszlo."  # uncited: the gate withholds it, it is not a refusal
    report = s.answer("Who plays Victor Laszlo?")
    assert report.failed and [c.verified for c in report.claims] == [False]


def test_config_partial_answers_off_by_default():
    assert load_config(CONFIG)["generator"]["partial_answers"] is False


def test_use_corpus_swaps_chunks_and_index_only(tmp_path):
    import json
    import os
    fork = tmp_path / "poisoned"
    fork.mkdir()
    (fork / "chunks.jsonl").write_text(json.dumps({"id": "alien-c03", "text": "Druillet designed the creature."}) + "\n",
                                       encoding="utf-8")
    calls = []

    class SwappableRetriever(FakeRetriever):
        def use(self, chunks, index_dir, rebuild=False):
            calls.append(([c.id for c in chunks], str(index_dir), rebuild))

    s = _sear()
    s.chunks_path, s.index_root = Path("corpus/chunks.jsonl"), tmp_path / "index"
    s.retriever = SwappableRetriever()
    generator, verifier = s.generator, s.verifier
    s.use_corpus(fork)
    assert s.by_id["alien-c03"].text == "Druillet designed the creature." and s.chunks_path == fork / "chunks.jsonl"
    assert calls == [(["alien-c03"], str(tmp_path / "index" / "poisoned"), True)]  # no index yet: build it
    assert s.generator is generator and s.verifier is verifier  # no model reloaded
    s.use_corpus(fork)
    assert len(calls) == 1  # already in use: nothing to do
    index = tmp_path / "index" / "poisoned" / "chunks.faiss"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"")
    s.chunks_path = Path("corpus/chunks.jsonl")
    s.use_corpus(fork)
    assert calls[-1][2] is False  # a fresh index is reused
    os.utime(index, (0, 0))
    s.chunks_path = Path("corpus/chunks.jsonl")
    s.use_corpus(fork)
    assert calls[-1][2] is True  # an index older than its chunks is rebuilt


def test_corpora_lists_the_main_corpus_and_its_forks():
    from sear_demo.corpus import corpora
    found = corpora(Path(__file__).resolve().parents[1] / "corpus")
    assert list(found)[0] == "corpus" and {"poisoned", "conflict"} <= set(found)


def test_retriever_use_rebuilds_an_index_of_the_wrong_size(tmp_path):
    pytest.importorskip("faiss")
    retrieve = importlib.import_module("sear_demo.retrieve")
    if not hasattr(retrieve, "faiss"):
        pytest.skip("retrieve is stubbed: its GPU-stack imports are missing")
    import numpy as np

    class Embedder:
        def encode(self, texts, normalize_embeddings=True):
            return np.eye(4, dtype="float32")[[len(t) % 4 for t in texts]]

    r = retrieve.Retriever.__new__(retrieve.Retriever)
    r.model = Embedder()
    r.use(CHUNKS, tmp_path)
    assert r.index.ntotal == 2
    more = CHUNKS + [Chunk("jaws-c02", "Quint's boat is the Orca.")]
    r.use(more, tmp_path)  # the saved index holds 2 chunks, not 3: rebuilt, not reused
    assert r.index.ntotal == 3 and r.chunks == more
