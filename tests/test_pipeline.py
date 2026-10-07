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
