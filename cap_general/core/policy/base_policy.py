"""Base classes for policies."""

from abc import abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, ClassVar, Optional

from cap_general.core.base import RegisteredBase


@dataclass
class PolicyResult:
    """Result from policy inference."""

    code: str
    policy_name: str
    metadata: Optional[dict] = None


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
class PolicyBaseConfig:
    """Configuration for constructing a policy."""

    policy_type: str = "base"


class PolicyBase(RegisteredBase):
    """Abstract base class for policies."""

    _registry: ClassVar[dict[str, type["PolicyBase"]]] = {}
    config_cls: ClassVar[type[PolicyBaseConfig]] = PolicyBaseConfig
    registry_key_method: ClassVar[str] = "policy_type"

    @classmethod
    def policy_type(cls) -> str:
        """Return the registry key for this policy."""
        return "base"

    @classmethod
    def from_config(cls, config: "PolicyBaseConfig | dict[str, Any]") -> "PolicyBase":
        """Instantiate a registered policy from config."""
        config_data = cls._normalize_component_config(config, "policy_type")
        policy_type = config_data.pop("policy_type")
        policy_cls = cls.get_registered_type(policy_type)
        if policy_cls is None:
            raise KeyError(f"Unknown policy type: {policy_type}")

        config_cls = getattr(policy_cls, "config_cls", None)
        if config_cls is not None and is_dataclass(config_cls):
            config_obj = cls._build_dataclass_config(config_cls, config_data)
            return policy_cls(config=config_obj)
        return policy_cls(**config_data)

    @staticmethod
    def _normalize_component_config(
        config: "PolicyBaseConfig | dict[str, Any]",
        type_key: str,
    ) -> dict[str, Any]:
        if is_dataclass(config):
            return dict(config.__dict__)
        if not isinstance(config, dict):
            raise TypeError(f"Expected config dict, got {type(config).__name__}")

        data = dict(config.get("config", {}))
        for key, value in config.items():
            if key != "config":
                data[key] = value
        if "type" in data and type_key not in data:
            data[type_key] = data.pop("type")
        if type_key not in data:
            raise KeyError(f"Missing config field: {type_key}")
        return data

    @staticmethod
    def _build_dataclass_config(config_cls, config_data: dict[str, Any]):
        field_names = {field.name for field in fields(config_cls)}
        values = {key: value for key, value in config_data.items() if key in field_names}
        return config_cls(**values)

    @abstractmethod
    def inference(self, *args: Any, **kwargs: Any) -> Any:
        """Run local model inference."""
        pass

    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Return the name of the policy."""
        pass
