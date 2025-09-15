
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

# Avoid importing heavy optional deps at module import time
AutoModelForCausalLM = None  # type: ignore
AutoTokenizer = None  # type: ignore
PeftModel = None  # type: ignore


class ModelLoader:
    """Load and manage quantized models with optional LoRA adapters."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        lora_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.lora_path = lora_path
        self.model: Optional[nn.Module] = None
        self.tokenizer = None

    def load_model(self) -> nn.Module:
        """Return a loaded model instance, falling back to a stub."""
        if self.model is not None:
            return self.model

        global AutoModelForCausalLM, AutoTokenizer
        # Lazy import transformers only when needed
        if AutoModelForCausalLM is None:
            try:
                from transformers import AutoModelForCausalLM as _AutoModelForCausalLM, AutoTokenizer as _AutoTokenizer
                AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore
                AutoTokenizer = _AutoTokenizer  # type: ignore
            except Exception:
                AutoModelForCausalLM = None  # type: ignore
                AutoTokenizer = None  # type: ignore

        if AutoModelForCausalLM is None:
            # Fallback tiny linear model if transformers unavailable
            self.model = nn.Linear(10, 10)
        else:
            try:
                kwargs = {}
                if self.load_in_4bit:
                    kwargs["load_in_4bit"] = True
                if self.load_in_8bit:
                    kwargs["load_in_8bit"] = True
                kwargs["device_map"] = "auto"
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)  # type: ignore
                if AutoTokenizer is not None:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # type: ignore
            except Exception:
                # Fallback tiny linear model
                self.model = nn.Linear(10, 10)

        if self.lora_path:
            self.swap_lora(self.lora_path)
        self.model.to(self.device)
        return self.model

    def swap_lora(self, adapter_path: str) -> None:
        """Load a LoRA adapter at runtime."""
        global PeftModel
        if PeftModel is None:
            try:
                from peft import PeftModel as _PeftModel
                PeftModel = _PeftModel  # type: ignore
            except Exception:
                return
        if self.model is None:
            self.load_model()
        try:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)  # type: ignore
            self.lora_path = adapter_path
        except Exception:
            pass

    def save_state(self, output_dir: str) -> None:
        """Persist model weights for later restoration."""
        if self.model is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        torch.save({"state": self.model.state_dict(), "lora": self.lora_path}, os.path.join(output_dir, "model.pt"))

    def load_state(self, output_dir: str) -> None:
        """Load model weights from ``output_dir``."""
        path = os.path.join(output_dir, "model.pt")
        if not os.path.exists(path) or self.model is None:
            return
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data.get("state", {}))
        lora = data.get("lora")
        if lora:
            self.swap_lora(lora)
