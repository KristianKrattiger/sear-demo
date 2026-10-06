from .attention import attach_attention
from .corpus import load_chunks
from .gate import gate
from .generate import Generator
from .narrator import narrate
from .prompts import GENERATE, format_sources
from .retrieve import Retriever
from .segment import segment
from .schema import Report, Result
from .verify import Verifier


class Sear:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.chunks = load_chunks(cfg["corpus"]["chunks"])
        self.by_id = {c.id: c for c in self.chunks}
        self.retriever = Retriever(self.chunks, cfg["retrieval"]["embedder"], cfg["corpus"]["index_dir"])
        g = cfg["generator"]
        self.generator = Generator(g["model"], g["load_in_4bit"], g["max_new_tokens"], g["log_attention"])
        self.verifier = Verifier(cfg["verifier"]["model"], cfg["verifier"]["tau"])
        self.narrator_llm = self.generator  # swap for an API wrapper if narrator.backend == "api"

    def ask(self, query: str, narrator: bool | None = None) -> Result:
        retrieved = self.retriever.search(query, self.cfg["retrieval"]["top_k"])
        text, trace = self.generator(GENERATE.format(sources=format_sources(retrieved), query=query))
        report = Report(query=query, retrieved=retrieved, claims=[], raw=text)

        if text.strip().upper().startswith("NOT SUPPORTED"):
            report.failed = True
        else:
            claims = segment(text)
            allowed = {c.id: c for c in retrieved}
            self.verifier.check(claims, allowed)
            attach_attention(claims, trace, chunk_spans=None)
            report.claims = claims
            report.failed = not gate(claims, self.cfg["gate"]["phi"])
        if report.failed:
            report.failure_message = self.cfg["gate"]["failure_message"]

        use_narrator = self.cfg["narrator"]["enabled"] if narrator is None else narrator
        narration = narrate(report, self.narrator_llm, self.verifier, self.by_id) if use_narrator else None
        return Result(report=report, narration=narration)
