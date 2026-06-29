"""Shared base helpers for core CAP components."""

from abc import ABC
from dataclasses import fields, is_dataclass
from typing import Any, ClassVar, Dict, Type, get_type_hints


class RegisteredBase(ABC):
    """Base class mixin that provides decorator-based class registration."""

    _registry: ClassVar[Dict[str, Type["RegisteredBase"]]] = {}
    registry_key_method: ClassVar[str] = "registered_type"
    name: ClassVar[str] = ""

    @classmethod
    def register(cls):
        """Register a subclass using its registry key class method."""

        def decorator(subclass: Type["RegisteredBase"]) -> Type["RegisteredBase"]:
            registry_key_method = subclass.registry_key_method
            key_method = getattr(subclass, registry_key_method, None)
            if key_method is None or not callable(key_method):
                raise TypeError(
                    f"Subclass {subclass.__name__} must define "
                    f"{registry_key_method}() class method"
                )
            if not getattr(subclass, "name", None):
                raise TypeError(
                    f"Subclass {subclass.__name__} must define name class attribute"
                )

            subclass._registry[key_method()] = subclass
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
        config_obj = cls._build_dataclass_config(config_cls, config_data)
        return subclass(config=config_obj, **kwargs)

    @staticmethod
    def _build_dataclass_config(config_cls, config_data: dict[str, Any]):
        config_fields = fields(config_cls)
        field_names = {field.name for field in config_fields}
        if "reset_mode" in field_names and "reset_frequency" not in field_names:
            config_data = dict(config_data)
            legacy_reset_frequency = config_data.pop("reset_frequency", None)
            if legacy_reset_frequency is not None:
                config_data.setdefault("reset_mode", legacy_reset_frequency)
        type_hints = get_type_hints(config_cls)
        values = {
            key: RegisteredBase._coerce_dataclass_field(type_hints.get(key), value)
            for key, value in config_data.items()
            if key in field_names
        }
        return config_cls(**values)

    @staticmethod
    def _coerce_dataclass_field(field_type: Any, value: Any) -> Any:
        """Build nested dataclass fields from dictionaries when the annotation is concrete."""
        if isinstance(value, dict) and "type" in value:
            return value
        if isinstance(value, dict) and isinstance(field_type, type) and is_dataclass(field_type):
            return RegisteredBase._build_dataclass_config(field_type, value)
        return value
