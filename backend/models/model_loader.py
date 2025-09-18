import os, torch
from typing import Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from huggingface_hub import InferenceClient

class StreamLLM:
    """Adapter exposing .stream(prompt) -> iterator[str] for both local Transformers and HF Inference API."""
    def __init__(self):
        self.mode = os.getenv("HF_MODE","transformers")
        self.model_id = os.getenv("HF_MODEL_ID","Qwen/Qwen2.5-7B-Instruct")
        self.token = os.getenv("HF_TOKEN") or None
        if self.mode == "inference_api":
            self.client = InferenceClient(model=self.model_id, token=self.token)
            self.backend = "hf-inference-api"
        else:
            self._load_local()
            self.backend = "transformers"

    def _load_local(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if device=="mps" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto"
        )
        self.device = device

    def stream(self, prompt: str):
        if self.mode == "inference_api":
            gen = self.client.text_generation(
                prompt, stream=True,
                max_new_tokens=int(os.getenv("LLM_MAX_TOKENS","512")),
                temperature=0.2, top_p=0.9, return_full_text=False
            )
            for ev in gen:
                # ev may be TextGenerationResponseStream
                text = getattr(getattr(ev, "token", None), "text", None)
                if text is None:
                    text = str(ev)
                yield text
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=int(os.getenv("LLM_MAX_TOKENS","512")),
                do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.1
            )
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            for text in streamer:
                yield text

def load_llm():
    return StreamLLM()

def model_info():
    """Return a dict with model metadata for provenance."""
    mode = os.getenv("HF_MODE","transformers")
    mid = os.getenv("HF_MODEL_ID","Qwen/Qwen2.5-7B-Instruct")
    return {
        "model_name": mid,
        "backend": "hf-inference-api" if mode=="inference_api" else "transformers",
        "quantization": None  # not applicable for transformers default
    }
