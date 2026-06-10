"""Local vLLM policy implementation."""

from typing import Any

from cap_general.core.policy.base_policy import (
    PolicyResult,
    PolicyBase,
    normalize_prompt,
)


@PolicyBase.register()
class VLLMPolicy(PolicyBase):
    """Local in-process vLLM policy."""

    name = "vLLM Policy"

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        download_dir: str | None = None,
        trust_remote_code: bool = False,
        enforce_eager: bool | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float | None = None,
        stop: list[str] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        sampling_kwargs: dict[str, Any] | None = None,
    ):
        self._model_path = model_path
        self._dtype = dtype
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._download_dir = download_dir
        self._trust_remote_code = trust_remote_code
        self._enforce_eager = enforce_eager
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._stop = stop
        self._llm_kwargs = llm_kwargs or {}
        self._sampling_kwargs = sampling_kwargs or {}
        self._llm = None
        self._sampling_params_cls = None

    @classmethod
    def policy_type(cls) -> str:
        return "vllm"

    def _load_model(self):
        """Lazily load vLLM locally."""
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vllm is required for VLLMPolicy. Install it with: pip install vllm"
            ) from exc

        kwargs = {
            "model": self._model_path,
            "dtype": self._dtype,
            "tensor_parallel_size": self._tensor_parallel_size,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "trust_remote_code": self._trust_remote_code,
            **self._llm_kwargs,
        }
        if self._max_model_len is not None:
            kwargs["max_model_len"] = self._max_model_len
        if self._download_dir is not None:
            kwargs["download_dir"] = self._download_dir
        if self._enforce_eager is not None:
            kwargs["enforce_eager"] = self._enforce_eager

        self._llm = LLM(**kwargs)
        self._sampling_params_cls = SamplingParams

    def _format_prompt(self, prompt: str | list[dict[str, Any]]) -> str:
        normalized = normalize_prompt(prompt)
        if isinstance(normalized, str):
            return normalized

        tokenizer = self._llm.get_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("Chat message prompts require a tokenizer chat template")
        return tokenizer.apply_chat_template(
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
        stop: list[str] | None = None,
        **sampling_kwargs: Any,
    ) -> PolicyResult:
        """Run local vLLM inference."""
        self._load_model()
        formatted_prompt = self._format_prompt(prompt)

        resolved_temperature = (
            self._temperature if temperature is None else temperature
        )
        resolved_top_p = self._top_p if top_p is None else top_p
        resolved_stop = self._stop if stop is None else stop
        kwargs = {
            "max_tokens": self._max_new_tokens
            if max_new_tokens is None
            else max_new_tokens,
            "temperature": resolved_temperature,
            **self._sampling_kwargs,
            **sampling_kwargs,
        }
        if resolved_top_p is not None:
            kwargs["top_p"] = resolved_top_p
        if resolved_stop:
            kwargs["stop"] = resolved_stop

        sampling_params = self._sampling_params_cls(**kwargs)
        outputs = self._llm.generate([formatted_prompt], sampling_params)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return PolicyResult(
            code=text,
            policy_name=self.policy_name,
            metadata={
                "backend": "vllm",
                "model_path": self._model_path,
                "max_new_tokens": kwargs["max_tokens"],
                "temperature": resolved_temperature,
                "top_p": resolved_top_p,
            },
        )

    @property
    def policy_name(self) -> str:
        return f"VLLMPolicy({self._model_path})"
