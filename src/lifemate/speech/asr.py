"""Transformer-backed ASR model loading and batched inference."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any

import numpy as np

from lifemate.speech.preprocessing import load_wav


def load_asr_config(path: str | Path = "configs/asr.yaml") -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Loading YAML configuration requires PyYAML") from exc
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source) or {}
    if not isinstance(config, dict):
        raise ValueError(f"ASR configuration must be a mapping: {path}")
    return config


def load_runtime_config(path: str | Path = "configs/runtime.yaml") -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Loading YAML configuration requires PyYAML") from exc
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Runtime configuration must be a mapping: {path}")
    return config


class ASRModel:
    """Loads a Hugging Face Whisper or CTC checkpoint lazily on construction."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.preprocessing = dict(self.config.get("preprocessing", {}))
        self.runtime = dict(self.config.get("runtime", {}))

        # Dynamically extract model config from either "model" or the first entry in "models" list
        model_config = dict(self.config.get("model", {}))
        if not model_config and "models" in self.config and isinstance(self.config["models"], list) and self.config["models"]:
            model_config = dict(self.config["models"][0])

        self.model_name = str(model_config.get("model_name_or_path", "openai/whisper-small"))

        cache_dir_env = os.environ.get("HF_HOME")
        self.cache_dir = cache_dir_env or model_config.get("cache_dir", "models/asr")

        offline_env = os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"
        self.local_files_only = offline_env or bool(model_config.get("local_files_only", False))

        self.language = model_config.get("language", "vietnamese")
        self.task = model_config.get("task", "transcribe")
        self.max_new_tokens = int(model_config.get("max_new_tokens", 128))
        self.sample_rate = int(self.preprocessing.get("sample_rate", 16_000))
        if self.sample_rate <= 0:
            raise ValueError("preprocessing.sample_rate must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("model.max_new_tokens must be positive")

        try:
            import torch
            from transformers import AutoConfig, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "ASR inference requires PyTorch and transformers; install the 'asr' optional dependencies."
            ) from exc

        self._torch = torch
        self.device = self._resolve_device(torch, self.runtime.get("device", "auto"))
        self.dtype = self._resolve_dtype(torch, self.runtime.get("precision", "auto"), self.device)
        options = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        try:
            model_config = AutoConfig.from_pretrained(self.model_name, **options)
            self.processor = AutoProcessor.from_pretrained(self.model_name, **options)
            if getattr(model_config, "model_type", "") == "whisper":
                self.architecture = "seq2seq"
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name, torch_dtype=self.dtype, **options
                )
            else:
                self.architecture = "ctc"
                self.model = AutoModelForCTC.from_pretrained(
                    self.model_name, torch_dtype=self.dtype, **options
                )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ASR model {self.model_name!r}; check the checkpoint, cache, and model access."
            ) from exc
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(torch, requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
        return requested

    @staticmethod
    def _resolve_dtype(torch, requested: str, device: str):
        names = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if requested == "auto":
            return torch.float16 if device.startswith("cuda") else torch.float32
        if requested not in names:
            raise ValueError(f"Unsupported ASR precision: {requested}")
        if requested == "float16" and device == "cpu":
            raise ValueError("float16 inference on CPU is not supported by this pipeline")
        return names[requested]

    def _load_audio(self, path: str | Path) -> np.ndarray:
        audio = load_wav(
            path,
            target_sample_rate=self.sample_rate,
            mono=bool(self.preprocessing.get("mono", True)),
        )
        if audio.samples.ndim != 1:
            raise ValueError("ASR model expects mono audio; set preprocessing.mono=true")
        return audio.samples

    def transcribe(
        self,
        audio: str | Path,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        return self.transcribe_batch([audio], contexts=[context])[0]

    def transcribe_batch(
        self,
        audio_paths: Sequence[str | Path],
        contexts: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[str]:
        if not audio_paths:
            return []
        if contexts is None:
            contexts = [None] * len(audio_paths)
        if len(contexts) != len(audio_paths):
            raise ValueError("contexts and audio_paths must have the same length")
        if self.architecture == "seq2seq" and len(audio_paths) > 1:
            prompts = [self._context_prompt(context) for context in contexts]
            if any(prompts) and len(set(prompts)) > 1:
                return [
                    self.transcribe(path, context=context)
                    for path, context in zip(audio_paths, contexts)
                ]
        audio = [self._load_audio(path) for path in audio_paths]
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            if self.architecture == "seq2seq":
                generation_args = {
                    **inputs,
                    "max_new_tokens": self.max_new_tokens,
                    "language": self.language,
                    "task": self.task,
                }
                prompt_ids = self._prompt_ids(contexts, len(audio))
                if prompt_ids is not None:
                    generation_args["prompt_ids"] = prompt_ids
                outputs = self.model.generate(**generation_args)
                transcripts = self.processor.batch_decode(outputs, skip_special_tokens=True)
            else:
                logits = self.model(**inputs).logits
                transcripts = self.processor.batch_decode(logits.argmax(dim=-1))
        return [text.strip() for text in transcripts]

    def _prompt_ids(self, contexts: Sequence[Mapping[str, Any] | None], batch_size: int):
        prompts = [self._context_prompt(context) for context in contexts]
        if not any(prompts) or not hasattr(self.processor, "get_prompt_ids"):
            return None
        token_ids = [
            self.processor.get_prompt_ids(prompt, return_tensors="pt")
            if prompt
            else self.processor.get_prompt_ids(" ", return_tensors="pt")
            for prompt in prompts
        ]
        if len({tuple(item.shape) for item in token_ids}) != 1:
            return None
        return self._torch.cat(token_ids, dim=0).to(self.device).reshape(batch_size, -1)

    @staticmethod
    def _context_prompt(context: Mapping[str, Any] | None) -> str:
        if not isinstance(context, Mapping):
            return ""
        prompt = context.get("prompt")
        hotwords = context.get("hotwords", [])
        parts = [str(prompt)] if isinstance(prompt, str) and prompt.strip() else []
        if isinstance(hotwords, Sequence) and not isinstance(hotwords, (str, bytes)):
            parts.extend(str(word) for word in hotwords if str(word).strip())
        return ", ".join(parts)
