from pathlib import Path

from .attention import attach_attention
from .corpus import load_chunks
from .gate import gate
from .generate import Generator
from .narrator import narrate
from .prompts import GENERATE, GENERATE_PARTIAL, format_sources
from .retrieve import Retriever
from .segment import answer_and_declines, segment, sentence_claims
from .schema import Narration, Report, Result
from .verify import Verifier, corpus_names


class Sear:
    def __init__(self, cfg: dict):
        backend = cfg["narrator"].get("backend", "local")
        if backend != "local":  # fail before loading any model
            raise NotImplementedError(f"narrator backend {backend!r} is not wired yet")
        if cfg["verifier"].get("granularity", "claim") not in ("claim", "sentence"):
            raise ValueError(f"verifier.granularity must be claim or sentence, not {cfg['verifier']['granularity']!r}")
        self.cfg = cfg
        self.chunks_path, self.index_root = Path(cfg["corpus"]["chunks"]), Path(cfg["corpus"]["index_dir"])
        self.chunks = load_chunks(self.chunks_path)
        self.by_id = {c.id: c for c in self.chunks}
        self.retriever = Retriever(self.chunks, cfg["retrieval"]["embedder"], cfg["corpus"]["index_dir"])
        g = cfg["generator"]
        self.generator = Generator(g["model"], g["load_in_4bit"], g["max_new_tokens"], g["log_attention"])
        self.verifier = Verifier(cfg["verifier"]["model"], cfg["verifier"]["tau"])
        self.verifier.names = corpus_names(c.text for c in self.chunks)
        self.narrator_llm = self.generator  # local backend reuses the loaded generator

    def use_corpus(self, folder: str | Path) -> None:
        """Answer from the corpus in `folder` (its chunks.jsonl) from now on, without reloading any model.

        Only the chunks and the index change: the index lives in index_dir/<folder name>/ and is rebuilt
        when it is missing or older than the chunks. The embedder, generator and verifier stay loaded."""
        chunks_path = Path(folder) / "chunks.jsonl"
        if chunks_path == self.chunks_path:
            return
        index_dir = self.index_root / Path(folder).name
        index = index_dir / "chunks.faiss"
        stale = not index.exists() or index.stat().st_mtime < chunks_path.stat().st_mtime
        self.chunks = load_chunks(chunks_path)
        self.by_id = {c.id: c for c in self.chunks}
        self.retriever.use(self.chunks, index_dir, rebuild=stale)
        self.verifier.names = corpus_names(c.text for c in self.chunks)
        self.chunks_path = chunks_path

    def answer(self, query: str) -> Report:
        """Retrieve, generate, verify and gate. The report is final before any narration."""
        retrieved = self.retriever.search(query, self.cfg["retrieval"]["top_k"])
        partial = self.cfg["generator"].get("partial_answers", False)
        prompt = GENERATE_PARTIAL if partial else GENERATE
        text, trace = self.generator(prompt.format(sources=format_sources(retrieved), query=query))
        report = Report(query=query, retrieved=retrieved, claims=[], raw=text)

        if partial:  # answer the supported part; declined parts are shown, never verified, and not counted by phi
            claims, report.declined = answer_and_declines(text)
            report.failed = not claims
        else:  # NOT SUPPORTED anywhere fails the whole report
            report.failed = "NOT SUPPORTED" in text.upper()
            claims = [] if report.failed else segment(text)
        if not report.failed:
            v = self.cfg["verifier"]
            if v.get("granularity", "claim") == "sentence":  # verify, keep or drop each sentence on its own
                films = {c.id: c.meta.get("film") for c in retrieved}
                claims = sentence_claims(claims, v.get("sentence_context", "none"), films)
            allowed = {c.id: c for c in retrieved}
            self.verifier.check(claims, allowed)
            if self.cfg["verifier"].get("repair_citations", True):
                self.verifier.repair(claims, retrieved)
            attach_attention(claims, trace, chunk_spans=None)
            report.claims = claims
            report.failed = not gate(claims, self.cfg["gate"]["phi"])
        if report.failed:
            report.failure_message = self.cfg["gate"]["failure_message"]
        return report

    def narrate(self, report: Report) -> Narration:
        """Narrate an existing report without regenerating it, so the UI can flip the narrator."""
        n = self.cfg["narrator"]
        return narrate(report, self.narrator_llm, self.verifier, self.by_id,
                       min_claims=n["min_claims"], contradict_tau=n["contradict_tau"])

    def ask(self, query: str, narrator: bool | None = None) -> Result:
        report = self.answer(query)
        use = self.cfg["narrator"]["enabled"] if narrator is None else narrator
        return Result(report=report, narration=self.narrate(report) if use else None)
