from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, os
from backend.logging_setup import setup_logging
logger = setup_logging("llm")


GENERATOR_MODEL = os.getenv("GENERATOR_MODEL","Qwen/Qwen2.5-1.5B-Instruct")


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


class HFStreamer:
    def __init__(self, model_name: str = GENERATOR_MODEL):
        self.device = pick_device()
        logger.info("Loading generator on %s: %s", self.device, model_name)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.device in ("cuda","mps"): self.model.to(self.device)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tok, device=0 if self.device=="cuda" else -1)


    def stream(self, prompt: str):
        out = self.pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.0, pad_token_id=self.tok.eos_token_id)[0]["generated_text"]
        # yield once to match your .stream() consumption pattern
        yield out[len(prompt):]