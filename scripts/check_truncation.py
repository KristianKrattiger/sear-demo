"""Does the NLI verifier read the tail of a long joint premise, or is it cut at 512 tokens?

    python scripts/check_truncation.py

Joins the four Alien chunks the way Verifier.joint_premise does (about 600 tokens) and scores
a claim that only the last chunk supports. If the pipeline truncated at 512, that chunk would be
cut and entailment would collapse; the second line forces that truncation to show the difference.
"""
from sear_demo.config import load_config
from sear_demo.corpus import load_chunks
from sear_demo.verify import Verifier

cfg = load_config()
v = Verifier(cfg["verifier"]["model"], cfg["verifier"]["tau"])
alien = [c for c in load_chunks(cfg["corpus"]["chunks"]) if c.id.startswith("alien-")]
premise = v.joint_premise(alien)  # alien-c04, which holds the release year, comes last
claim = "Alien was released in 1983."
pair = {"text": premise, "text_pair": claim}

entail = lambda scores: next(s["score"] for s in scores if s["label"].lower().startswith("entail"))
model_cfg = v.nli.model.config
print(f"tokens the model receives:        {v.nli.preprocess(pair)['input_ids'].shape[-1]} "
      f"(tokenizer max {v.nli.tokenizer.model_max_length}, position_biased_input={getattr(model_cfg, 'position_biased_input', None)})")
print(f"entailment, full premise:         {entail(v.nli(pair)):.3f}")
print(f"entailment, premise cut to 512:   {entail(v.nli(pair, truncation='only_first', max_length=512)):.3f}")
