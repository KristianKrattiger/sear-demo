import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Generator:
    """Local HF model. Records attention over retrieved-chunk spans for the Stage 1c baseline."""

    def __init__(self, model: str, load_in_4bit: bool = True, max_new_tokens: int = 400, log_attention: bool = True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if load_in_4bit else None
        self.tok = AutoTokenizer.from_pretrained(model)
        # Whole model on GPU 0: fail loudly on OOM rather than silently offloading to CPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="eager" if log_attention else None,
        )
        self.max_new_tokens = max_new_tokens
        self.log_attention = log_attention

    def __call__(self, prompt: str) -> tuple[str, dict]:
        # Instruct models expect their chat template; raw prompts degrade format-following.
        chat = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tok(chat, return_tensors="pt", add_special_tokens=False).to(self.model.device)
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
