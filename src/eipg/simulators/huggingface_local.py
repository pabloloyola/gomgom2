"""Local Hugging Face Transformers chat client for EIPG experiments.

This adapter implements the same small chat interface used by the existing
OpenAI-compatible client, but performs inference directly in-process with a
Transformers model. Heavy optional dependencies are imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import time


@dataclass(frozen=True)
class HuggingFaceLocalConfig:
    model: str = "google/gemma-3-12b-it"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float | None = None
    cache_dir: str | None = ".cache/eipg/hf_local"
    trust_remote_code: bool = False
    model_loader: str = "auto"
    chat_template_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "HuggingFaceLocalConfig":
        return cls(
            model=str(section.get("model", "google/gemma-3-12b-it")),
            dtype=str(section.get("dtype", "bfloat16")),
            device_map=str(section.get("device_map", "auto")),
            max_new_tokens=int(section.get("max_new_tokens", 128)),
            do_sample=bool(section.get("do_sample", False)),
            temperature=(
                None
                if section.get("temperature") is None
                else float(section.get("temperature"))
            ),
            cache_dir=(str(section["cache_dir"]) if section.get("cache_dir") else None),
            trust_remote_code=bool(section.get("trust_remote_code", False)),
            model_loader=str(section.get("model_loader", "auto")),
            chat_template_kwargs=(
                dict(section["chat_template_kwargs"])
                if section.get("chat_template_kwargs")
                else None
            ),
        )


@dataclass(frozen=True)
class ChatResult:
    text: str
    cached: bool
    latency_seconds: float
    prompt_hash: str


def _select_loader_kind(
    *,
    requested: str,
    model_type: str,
    architectures: tuple[str, ...],
) -> str:
    """Resolve a reproducible HF loader family without model-name heuristics."""
    requested = requested.lower().strip()
    if requested not in {"auto", "causal", "multimodal"}:
        raise ValueError(
            "model_loader must be one of: auto, causal, multimodal; "
            f"got {requested!r}"
        )
    if requested != "auto":
        return requested

    multimodal_model_types = {"gemma3", "qwen3_5"}
    if model_type in multimodal_model_types:
        return "multimodal"
    if any(
        ("ConditionalGeneration" in name or "Multimodal" in name)
        and "CausalLM" not in name
        for name in architectures
    ):
        return "multimodal"
    return "causal"


class HuggingFaceLocalChatClient:
    """In-process text-only chat inference using Hugging Face Transformers."""

    def __init__(self, config: HuggingFaceLocalConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local HF inference requires the optional local-llm dependencies. "
                "Install with: pip install -e '.[local-llm]'"
            ) from exc

        dtype_name = config.dtype.lower()
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported dtype: {config.dtype}")

        hf_config = AutoConfig.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
        )
        model_type = str(getattr(hf_config, "model_type", "") or "")
        architectures = tuple(getattr(hf_config, "architectures", ()) or ())
        loader_kind = _select_loader_kind(
            requested=config.model_loader,
            model_type=model_type,
            architectures=architectures,
        )

        self._torch = torch
        self.transformers_version = str(transformers.__version__)
        self.model_type = model_type
        self.architectures = architectures
        self.loader_kind = loader_kind

        common_kwargs = {
            "trust_remote_code": config.trust_remote_code,
        }
        model_kwargs = {
            "torch_dtype": dtype_map[dtype_name],
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
        }

        if loader_kind == "multimodal":
            try:
                from transformers import AutoModelForMultimodalLM
                model_cls = AutoModelForMultimodalLM
            except ImportError:
                if model_type == "gemma3":
                    try:
                        from transformers import Gemma3ForConditionalGeneration
                    except ImportError as exc:
                        raise RuntimeError(
                            "This Gemma-3 installation needs a newer transformers build "
                            "with Gemma3ForConditionalGeneration or AutoModelForMultimodalLM."
                        ) from exc
                    model_cls = Gemma3ForConditionalGeneration
                else:
                    raise RuntimeError(
                        f"{config.model} is a multimodal model ({model_type=}) but the "
                        "installed transformers package has no AutoModelForMultimodalLM. "
                        "Upgrade transformers before running this model."
                    )
            self.processor = AutoProcessor.from_pretrained(config.model, **common_kwargs)
            self.model = model_cls.from_pretrained(config.model, **model_kwargs).eval()
        else:
            self.processor = AutoTokenizer.from_pretrained(config.model, **common_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model,
                **model_kwargs,
            ).eval()

    def runtime_info(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "loader_kind": self.loader_kind,
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "transformers_version": self.transformers_version,
            "processor_class": type(self.processor).__name__,
            "model_class": type(self.model).__name__,
            "chat_template_kwargs": self.config.chat_template_kwargs,
        }

    def _cache_key(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None,
    ) -> str:
        payload = {
            "backend": "huggingface_local",
            "model": self.config.model,
            "model_loader": self.loader_kind,
            "chat_template_kwargs": self.config.chat_template_kwargs,
            "dtype": self.config.dtype,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "messages": messages,
            "response_format": response_format,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(raw.encode("utf-8")).hexdigest()

    def _structured_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Normalize ordinary text chat messages for the selected HF chat template."""
        if self.loader_kind == "multimodal":
            return [
                {
                    "role": str(message["role"]),
                    "content": [{"type": "text", "text": str(message["content"])}],
                }
                for message in messages
            ]
        return [
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in messages
        ]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        key = self._cache_key(messages, response_format=response_format)
        path = None if self.cache_dir is None else self.cache_dir / f"{key}.json"
        if path is not None and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ChatResult(str(payload["text"]), True, 0.0, key)

        structured = self._structured_messages(messages)
        started = time.perf_counter()
        template_kwargs = dict(self.config.chat_template_kwargs or {})
        inputs = self.processor.apply_chat_template(
            structured,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )

        input_device = next(self.model.parameters()).device
        inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample and self.config.temperature is not None:
            generation_kwargs["temperature"] = self.config.temperature

        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        prompt_len = int(inputs["input_ids"].shape[-1])
        new_tokens = generated[:, prompt_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        if not text:
            raise RuntimeError("local Hugging Face model returned empty assistant content")

        latency = time.perf_counter() - started
        if path is not None:
            path.write_text(
                json.dumps(
                    {
                        "text": text,
                        "model": self.config.model,
                        "loader_kind": self.loader_kind,
                        "latency_seconds": latency,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return ChatResult(text, False, latency, key)
