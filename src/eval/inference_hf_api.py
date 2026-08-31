from __future__ import annotations

import os
import re
import time
from typing import Optional, Tuple

_BOOL_RE = re.compile(r'\b(true|false)\b', re.IGNORECASE)

def _parse_bool(text: str) -> Optional[bool]:
    m = _BOOL_RE.search(text)
    if not m:
        return None
    return m.group(1).lower() == 'true'

class HFInferenceClient:
    def __init__(self, model_id: str, token: Optional[str] = None, max_new_tokens: int = 8, max_retries: int = 5, base_delay: float = 1.0):
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("huggingface_hub is not installed. Please install it with `pip install huggingface_hub`.")
            
        if token is None:
            token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            
        self.model_id = model_id
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = InferenceClient(model=model_id, token=token)

    def _retry_wrapper(self, func, *args, **kwargs):
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise e

    def predict(self, prompt: str, temperature: float = 0.0) -> Tuple[Optional[bool], str]:
        def _call():
            return self.client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0
            )
        response = self._retry_wrapper(_call)
        return _parse_bool(response), response

    def predict_chat(self, prompt: str, temperature: float = 0.0) -> Tuple[Optional[bool], str]:
        def _call():
            return self.client.chat_completion(
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=self.max_new_tokens,
                temperature=temperature if temperature > 0 else None
            )
        response = self._retry_wrapper(_call)
        content = response.choices[0].message.content
        return _parse_bool(content), content

def create_inference_client(model_id: str, token: Optional[str] = None, use_chat: bool = True, max_new_tokens: int = 8) -> HFInferenceClient:
    return HFInferenceClient(model_id=model_id, token=token, max_new_tokens=max_new_tokens)
