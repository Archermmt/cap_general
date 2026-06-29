"""Hugging Face policy implementation."""

import logging
from dataclasses import dataclass, field
from typing import Any

from cap_general.core.policy.base_policy import BasePolicy, PolicyResult

from .base_policy import BasePolicyConfig


def apply_stop_sequences(text: str, stop: list[str] | None = None) -> str:
    """Truncate text at the earliest stop sequence."""
    if not stop:
        return text

    earliest = None
    for sequence in stop:
        if not sequence:
            continue
        index = text.find(sequence)
        if index >= 0 and (earliest is None or index < earliest):
            earliest = index

    return text if earliest is None else text[:earliest]


def normalize_prompt(prompt: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Normalize supported prompt inputs for local model backends."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return prompt
    raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


@dataclass
class HuggingFacePolicyConfig(BasePolicyConfig):
    """Configuration for HuggingFacePolicy."""

    model_path: str
    device: str = "auto"
    torch_dtype: str | None = "auto"
    device_map: str | dict[str, Any] | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    cache_dir: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float | None = None
    do_sample: bool | None = None
    stop: list[str] | None = None
    return_full_text: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    describe: str = field(
        default=(
            "Generates text or code from prompts with a local Hugging Face "
            "Transformers causal language model."
        ),
        kw_only=True,
    )


@BasePolicy.register()
class HuggingFacePolicy(BasePolicy):
    """A local Transformers policy loaded from a Hugging Face checkpoint."""

    name = "HuggingFace Policy"
    config_cls = HuggingFacePolicyConfig

    def __init__(
        self,
        config: HuggingFacePolicyConfig,
        logger: logging.Logger,
    ):
        """Initialize a local Transformers model."""
        super().__init__(config=config, logger=logger)
        self._model_path = config.model_path
        self._device = config.device
        self._torch_dtype = config.torch_dtype
        self._device_map = config.device_map
        self._trust_remote_code = config.trust_remote_code
        self._local_files_only = config.local_files_only
        self._cache_dir = config.cache_dir
        self._max_new_tokens = config.max_new_tokens
        self._temperature = config.temperature
        self._top_p = config.top_p
        self._do_sample = config.do_sample
        self._stop = config.stop
        self._return_full_text = config.return_full_text
        self._model_kwargs = config.model_kwargs
        self._tokenizer_kwargs = config.tokenizer_kwargs
        self._generation_kwargs = config.generation_kwargs
        self._model = None
        self._tokenizer = None
        self._torch = None

    @classmethod
    def policy_type(cls) -> str:
        return "huggingface"

    def reset(self) -> None:
        """Load the model and tokenizer if needed."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer_kwargs = {
                    "trust_remote_code": self._trust_remote_code,
                    "local_files_only": self._local_files_only,
                    **self._tokenizer_kwargs,
                }
                if self._cache_dir is not None:
                    tokenizer_kwargs["cache_dir"] = self._cache_dir

                model_kwargs = {
                    "trust_remote_code": self._trust_remote_code,
                    "local_files_only": self._local_files_only,
                    **self._model_kwargs,
                }
                if self._cache_dir is not None:
                    model_kwargs["cache_dir"] = self._cache_dir
                if self._torch_dtype is not None:
                    model_kwargs["torch_dtype"] = self._resolve_torch_dtype(torch)
                if self._device_map is not None:
                    model_kwargs["device_map"] = self._device_map

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_path, **tokenizer_kwargs
                )
                self._model = AutoModelForCausalLM.from_pretrained(self._model_path, **model_kwargs)
                if self._device_map is None and self._device != "auto":
                    self._model.to(self._device)
                self._model.eval()
                if (
                    self._tokenizer.pad_token_id is None
                    and self._tokenizer.eos_token_id is not None
                ):
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                self._torch = torch
            except ImportError:
                raise ImportError(
                    "transformers library is required for HuggingFacePolicy. "
                    "Install it with: pip install transformers torch"
                )

    def _resolve_torch_dtype(self, torch):
        """Resolve dtype strings into torch dtype objects."""
        if self._torch_dtype == "auto":
            return "auto"
        if isinstance(self._torch_dtype, str):
            return getattr(torch, self._torch_dtype)
        return self._torch_dtype

    def _format_prompt(self, prompt: str | list[dict[str, Any]]) -> str:
        """Format plain prompts or chat messages for the tokenizer."""
        normalized = normalize_prompt(prompt)
        if isinstance(normalized, str):
            return normalized
        return self._tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )

    def inference(
        self,
        prompt: str | list[dict[str, Any]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
        stop: list[str] | None = None,
        **generation_kwargs: Any,
    ) -> PolicyResult:
        """Run local Transformers inference."""
        self.reset()

        formatted_prompt = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
        if self._device_map is None and self._device != "auto":
            inputs = inputs.to(self._device)

        resolved_temperature = self._temperature if temperature is None else temperature
        resolved_top_p = self._top_p if top_p is None else top_p
        if do_sample is not None:
            resolved_do_sample = do_sample
        elif self._do_sample is not None:
            resolved_do_sample = self._do_sample
        else:
            resolved_do_sample = resolved_temperature > 0

        kwargs = {
            "max_new_tokens": self._max_new_tokens if max_new_tokens is None else max_new_tokens,
            "do_sample": resolved_do_sample,
            **self._generation_kwargs,
            **generation_kwargs,
        }
        if resolved_temperature is not None and resolved_temperature > 0:
            kwargs["temperature"] = resolved_temperature
        if resolved_top_p is not None:
            kwargs["top_p"] = resolved_top_p
        if self._tokenizer.eos_token_id is not None:
            kwargs.setdefault("eos_token_id", self._tokenizer.eos_token_id)
        if self._tokenizer.pad_token_id is not None:
            kwargs.setdefault("pad_token_id", self._tokenizer.pad_token_id)

        with self._torch.inference_mode():
            outputs = self._model.generate(**inputs, **kwargs)

        if self._return_full_text:
            generated_tokens = outputs[0]
        else:
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_text = apply_stop_sequences(generated_text, stop or self._stop)

        return PolicyResult(
            code=generated_text,
            policy_name=self.policy_type(),
            metadata={
                "backend": "transformers",
                "model_path": self._model_path,
                "device": self._device,
                "max_new_tokens": kwargs["max_new_tokens"],
                "temperature": resolved_temperature,
                "top_p": resolved_top_p,
            },
        )
