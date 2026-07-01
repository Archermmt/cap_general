"""Shared base helpers for core CAP components."""

from abc import ABC
from dataclasses import is_dataclass
from typing import Any, ClassVar, Dict, Type

from cap_general.core.utils.config import build_dataclass_config


class RegisteredBase(ABC):
    """Base class mixin that provides decorator-based class registration."""

    _registry: ClassVar[Dict[str, Type["RegisteredBase"]]] = {}
    registry_key_attr: ClassVar[str] = "registered_type"

    @classmethod
    def register(cls):
        """Register a subclass using its registry key class attribute."""

        def decorator(subclass: Type["RegisteredBase"]) -> Type["RegisteredBase"]:
            registry_key_attr = subclass.registry_key_attr
            registry_key = getattr(subclass, registry_key_attr, None)
            if not isinstance(registry_key, str) or not registry_key:
                raise TypeError(
                    f"Subclass {subclass.__name__} must define "
                    f"{registry_key_attr} class attribute"
                )
            subclass._registry[registry_key] = subclass
            return subclass

        return decorator

    @classmethod
    def get_registered_class(cls, registered_type: str) -> Type["RegisteredBase"] | None:
        """Get a registered class by its registry key."""
        return cls._registry.get(registered_type)

    @classmethod
    def registered_types(cls) -> list[str]:
        """Return all registered type keys for this base class."""
        return list(cls._registry.keys())

    @classmethod
    def get_registry(cls) -> Dict[str, Type["RegisteredBase"]]:
        """Return a shallow copy of the current registry."""
        return dict(cls._registry)

    @classmethod
    def create(cls, registered_type: str, *args: Any, **kwargs: Any) -> "RegisteredBase":
        """Instantiate a registered class by type key."""
        subclass = cls.get_registered_class(registered_type)
        if subclass is None:
            raise KeyError(f"Unknown registered type: {registered_type}")
        return subclass(*args, **kwargs)

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "RegisteredBase":
        """Instantiate a registered subclass from config with a ``type`` field."""
        if is_dataclass(config):
            config_data = dict(config.__dict__)
        elif isinstance(config, dict):
            config_data = dict(config)
        else:
            raise TypeError(f"Expected config, got {type(config).__name__}")

        registered_type = config_data.pop("type")
        subclass = cls.get_registered_class(registered_type)
        if subclass is None:
            raise KeyError(f"Unknown registered type: {registered_type}")

        config_cls = getattr(subclass, "config_cls", None)
        if config_cls is None or not is_dataclass(config_cls):
            raise TypeError(f"Registered class {subclass.__name__} must define dataclass config_cls")
        config_obj = build_dataclass_config(config_cls, config_data)
        return subclass(config=config_obj, **kwargs)
