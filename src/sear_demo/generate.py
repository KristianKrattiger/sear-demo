import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Generator:
    """Local HF model. Records attention over retrieved-chunk spans for the Stage 1c baseline."""

    def __init__(self, model: str, load_in_4bit: bool = True, max_new_tokens: int = 400, log_attention: bool = True):
        quant = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=quant, device_map="auto",
            attn_implementation="eager" if log_attention else None,
        )
        self.max_new_tokens = max_new_tokens
        self.log_attention = log_attention

    def __call__(self, prompt: str) -> tuple[str, dict]:
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                output_attentions=self.log_attention, return_dict_in_generate=True,
            )
        new_tokens = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.tok.decode(new_tokens, skip_special_tokens=True)
        trace = {"attentions": out.attentions if self.log_attention else None,
                 "prompt_len": inputs["input_ids"].shape[1]}
        return text, trace
