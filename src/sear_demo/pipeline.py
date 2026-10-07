from .attention import attach_attention
from .corpus import load_chunks
from .gate import gate
from .generate import Generator
from .narrator import narrate
from .prompts import GENERATE, format_sources
from .retrieve import Retriever
from .segment import segment
from .schema import Narration, Report, Result
from .verify import Verifier


class Sear:
    def __init__(self, cfg: dict):
        backend = cfg["narrator"].get("backend", "local")
        if backend != "local":  # fail before loading any model
            raise NotImplementedError(f"narrator backend {backend!r} is not wired yet")
        self.cfg = cfg
        self.chunks = load_chunks(cfg["corpus"]["chunks"])
        self.by_id = {c.id: c for c in self.chunks}
        self.retriever = Retriever(self.chunks, cfg["retrieval"]["embedder"], cfg["corpus"]["index_dir"])
        g = cfg["generator"]
        self.generator = Generator(g["model"], g["load_in_4bit"], g["max_new_tokens"], g["log_attention"])
        self.verifier = Verifier(cfg["verifier"]["model"], cfg["verifier"]["tau"])
        self.narrator_llm = self.generator  # local backend reuses the loaded generator

    def answer(self, query: str) -> Report:
        """Retrieve, generate, verify and gate. The report is final before any narration."""
        retrieved = self.retriever.search(query, self.cfg["retrieval"]["top_k"])
        text, trace = self.generator(GENERATE.format(sources=format_sources(retrieved), query=query))
        report = Report(query=query, retrieved=retrieved, claims=[], raw=text)

        if "NOT SUPPORTED" in text.upper():
            report.failed = True
        else:
            claims = segment(text)
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
