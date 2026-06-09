"""Hugging Face policy model implementation."""

from cap_general.core.models.base_model import PolicyGenerationResult, PolicyModel


@PolicyModel.register()
class HuggingFacePolicyModel(PolicyModel):
    """A policy model that loads from a Hugging Face checkpoint."""

    name = "Hugging Face Policy Model"

    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize with model path."""
        self._model_path = model_path
        self._device = device
        self._model = None
        self._tokenizer = None

    @classmethod
    def model_type(cls) -> str:
        return "huggingface"

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
        """Generate code using the loaded model."""
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
