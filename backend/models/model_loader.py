import os, torch, logging
from typing import Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

logger = logging.getLogger("backend.model_loader")

class StreamLLM:
    """Adapter exposing .stream(prompt) -> iterator[str] for both local Transformers and HF Inference API."""
    def __init__(self):
        self.mode = os.getenv("HF_MODE","transformers")
        self.model_id = os.getenv("HF_MODEL_ID","Qwen/Qwen2.5-1.5B-Instruct")
        self.token = os.getenv("HF_TOKEN") or None
        self.logger = logger
        self.logger.info("StreamLLM mode=%s model_id=%s", self.mode, self.model_id)
        if self.mode == "inference_api":
            self.client = InferenceClient(model=self.model_id, token=self.token)
            self.backend = "hf-inference-api"
        elif self.mode == "router":
            if OpenAI is None:
                raise RuntimeError("openai package is required for router mode (pip install openai)")
            # router client uses HF_TOKEN as API key
            self.router = OpenAI(base_url="https://router.huggingface.co/v1", api_key=self.token)
            self.backend = "hf-router"
        else:
            self._load_local()
            self.backend = "transformers"

    def _load_local(self):
        self.logger.info("Loading local model %s", self.model_id)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if device=="mps" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto"
        )
        self.device = device
        self.logger.info("Loaded model on device=%s dtype=%s", device, dtype)

    def stream(self, prompt: str):
        if self.mode == "inference_api":
            try:
                gen = self.client.text_generation(
                    prompt, stream=True,
                    max_new_tokens=int(os.getenv("LLM_MAX_TOKENS","512")),
                    temperature=0.2, top_p=0.9, return_full_text=False
                )
            except HfHubHTTPError as e:
                # Provide clearer logging and guidance for common issues like 404 (model not found)
                self.logger.error("Hugging Face Inference API call failed for model %s: %s", self.model_id, str(e))
                raise RuntimeError(
                    f"Inference API request failed for model '{self.model_id}': {e}.\n"
                    "Common causes: model id is incorrect, model is private/gated and requires accepting terms, or your HF token is missing/insufficient.\n"
                    "Remedies: verify the model id on huggingface.co, accept model repo terms, set HF_TOKEN in backend/.env or login with `huggingface-cli login`, and ensure HF_MODE is 'inference_api'."
                ) from e

            try:
                for ev in gen:
                    # ev may be TextGenerationResponseStream
                    text = getattr(getattr(ev, "token", None), "text", None)
                    if text is None:
                        text = str(ev)
                    yield text
            except HfHubHTTPError as e:
                # If inference_api fails with 404/403, try router fallback when possible
                self.logger.error("Hugging Face Inference API streaming error for model %s: %s", self.model_id, str(e))
                if OpenAI is not None:
                    self.logger.info("Attempting router fallback for model %s", self.model_id)
                    # attempt router approach if available
                    try:
                        for t in self._stream_via_router(prompt):
                            yield t
                        return
                    except Exception as e2:
                        self.logger.error("Router fallback failed: %s", str(e2))
                raise RuntimeError(
                    f"Inference API streaming failed for model '{self.model_id}': {e}.\n"
                    "This may indicate a permissions issue or a transient error. Ensure your HF_TOKEN has access and consider retrying."
                ) from e
        elif self.mode == "router":
            # use router-based streaming exclusively
            for t in self._stream_via_router(prompt):
                yield t
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

    def _stream_via_router(self, prompt: str):
        """Stream text from Hugging Face OpenAI-compatible router using openai.OpenAI client."""
        if OpenAI is None:
            raise RuntimeError("openai package not available; install with pip install openai")
        # Build a simple chat message; some router models expect chat format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        model_id = self.model_id
        # If model id doesn't include a provider suffix, try using as-is
        stream = self.router.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS","512")),
            temperature=0.2,
            top_p=0.9,
        )
        for chunk in stream:
            # try attribute access first (openai SDK returns objects with .choices)
            text = None
            try:
                text = chunk.choices[0].delta.content
            except Exception:
                try:
                    text = chunk["choices"][0]["delta"].get("content")
                except Exception:
                    text = None
            if text:
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
