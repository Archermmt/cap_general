"""HuggingFace model operator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cap_general.core.operator.base_operator import BaseOperator, to_stage_fn
from cap_general.core.operator.model.base_model_op import ModelOp
from cap_general.core.policy.policy_result import PolicyResult


@dataclass
class HuggingFaceConfig:
    model_path: str = ""
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


@BaseOperator.register()
class HuggingFaceOp(ModelOp):
    """A local Transformers model loaded from a Hugging Face checkpoint."""

    op_type = "huggingface"
    config_cls = HuggingFaceConfig

    def reset(self) -> None:
        self._model = None
        self._tokenizer = None
        self._torch = None
        super().reset()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer_kwargs = {
                "trust_remote_code": self._config.trust_remote_code,
                "local_files_only": self._config.local_files_only,
                **self._config.tokenizer_kwargs,
            }
            if self._config.cache_dir is not None:
                tokenizer_kwargs["cache_dir"] = self._config.cache_dir
            model_kwargs = {
                "trust_remote_code": self._config.trust_remote_code,
                "local_files_only": self._config.local_files_only,
                **self._config.model_kwargs,
            }
            if self._config.cache_dir is not None:
                model_kwargs["cache_dir"] = self._config.cache_dir
            if self._config.torch_dtype is not None:
                model_kwargs["torch_dtype"] = self._resolve_torch_dtype(torch)
            if self._config.device_map is not None:
                model_kwargs["device_map"] = self._config.device_map
            self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_path, **tokenizer_kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(self._config.model_path, **model_kwargs)
            if self._config.device_map is None and self._config.device != "auto":
                self._model.to(self._config.device)
            self._model.eval()
            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._torch = torch
        except ImportError:
            raise ImportError("transformers library is required for HuggingFaceOp.")

    def _resolve_torch_dtype(self, torch):
        if self._config.torch_dtype == "auto":
            return "auto"
        if isinstance(self._config.torch_dtype, str):
            return getattr(torch, self._config.torch_dtype)
        return self._config.torch_dtype

    def _format_prompt(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            return prompt
        return self._tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    @to_stage_fn
    def inference(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._load_model()
        prompt = inputs["prompt"]
        max_new_tokens = inputs.get("max_new_tokens")
        temperature = inputs.get("temperature")
        top_p = inputs.get("top_p")
        do_sample = inputs.get("do_sample")
        stop = inputs.get("stop")
        extra_generation_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ("prompt", "max_new_tokens", "temperature", "top_p", "do_sample", "stop")
        }
        formatted_prompt = self._format_prompt(prompt)
        tokenizer_inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
        if self._config.device_map is None and self._config.device != "auto":
            tokenizer_inputs = tokenizer_inputs.to(self._config.device)
        resolved_temperature = self._config.temperature if temperature is None else temperature
        resolved_top_p = self._config.top_p if top_p is None else top_p
        if do_sample is not None:
            resolved_do_sample = do_sample
        elif self._config.do_sample is not None:
            resolved_do_sample = self._config.do_sample
        else:
            resolved_do_sample = resolved_temperature > 0
        kwargs = {
            "max_new_tokens": self._config.max_new_tokens if max_new_tokens is None else max_new_tokens,
            "do_sample": resolved_do_sample,
            **self._config.generation_kwargs,
            **extra_generation_kwargs,
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
            outputs = self._model.generate(**tokenizer_inputs, **kwargs)
        if self._config.return_full_text:
            generated_tokens = outputs[0]
        else:
            generated_tokens = outputs[0][tokenizer_inputs["input_ids"].shape[-1] :]
        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if stop or self._config.stop:
            for seq in stop or self._config.stop:
                idx = generated_text.find(seq)
                if idx >= 0:
                    generated_text = generated_text[:idx]
                    break
        return {
            "output": PolicyResult(
                code=generated_text,
                policy_name="huggingface",
                metadata={
                    "backend": "transformers",
                    "model_path": self._config.model_path,
                    "device": self._config.device,
                    "max_new_tokens": kwargs["max_new_tokens"],
                    "temperature": resolved_temperature,
                    "top_p": resolved_top_p,
                },
            )
        }
