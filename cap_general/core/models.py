"""Policy models for code generation."""

from abc import ABC, abstractmethod
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class PolicyGenerationResult:
    """Result from policy model code generation."""

    code: str
    model_name: str
    metadata: Optional[dict] = None


class PolicyModel(ABC):
    """Abstract base class for policy models that generate code from prompts."""

    @abstractmethod
    def generate(self, prompt: str) -> PolicyGenerationResult:
        """Generate code from a natural language prompt.

        Args:
            prompt: Natural language description of desired behavior.

        Returns:
            PolicyGenerationResult containing generated code.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model."""
        pass


class StaticPolicyModel(PolicyModel):
    """A policy model that always returns the same fixed code.

    Useful for testing and deterministic behavior.
    """

    def __init__(self, code: str):
        """Initialize with fixed code.

        Args:
            code: The fixed code to return on every generate() call.
        """
        self._code = code

    def generate(self, prompt: str) -> PolicyGenerationResult:
        """Return the fixed code regardless of prompt."""
        return PolicyGenerationResult(code=self._code, model_name=self.model_name)

    @property
    def model_name(self) -> str:
        return "StaticPolicyModel"


class CallablePolicyModel(PolicyModel):
    """A policy model that uses a callable function to generate code.

    Allows custom code generation logic to be injected.
    """

    def __init__(
        self,
        generator_fn: Callable[[str], str],
        model_name: str = "CallablePolicyModel",
    ):
        """Initialize with a generator function.

        Args:
            generator_fn: Function that takes a prompt and returns generated code.
            model_name: Name identifier for this model.
        """
        self._generator_fn = generator_fn
        self._model_name = model_name

    def generate(self, prompt: str) -> PolicyGenerationResult:
        """Generate code using the provided function."""
        code = self._generator_fn(prompt)
        return PolicyGenerationResult(code=code, model_name=self.model_name)

    @property
    def model_name(self) -> str:
        return self._model_name


class HuggingFacePolicyModel(PolicyModel):
    """A policy model that loads from a Hugging Face checkpoint.

    Lazily loads the model on first use to avoid unnecessary initialization.
    Requires transformers library.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize with model path.

        Args:
            model_path: Path to Hugging Face model or model identifier.
            device: Device to run model on ('cpu' or 'cuda').
        """
        self._model_path = model_path
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazily load the model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
                self._model = AutoModelForCausalLM.from_pretrained(self._model_path)
                self._model.to(self._device)
            except ImportError:
                raise ImportError(
                    "transformers library is required for HuggingFacePolicyModel. "
                    "Install it with: pip install transformers"
                )

    def generate(self, prompt: str, max_length: int = 512) -> PolicyGenerationResult:
        """Generate code using the loaded model.

        Args:
            prompt: Natural language prompt.
            max_length: Maximum length of generated sequence.

        Returns:
            PolicyGenerationResult with generated code.
        """
        self._load_model()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        outputs = self._model.generate(**inputs, max_length=max_length)
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        return PolicyGenerationResult(
            code=generated_text,
            model_name=self.model_name,
            metadata={"max_length": max_length, "device": self._device},
        )

    @property
    def model_name(self) -> str:
        return f"HuggingFacePolicyModel({self._model_path})"
