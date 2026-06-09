"""Shared base helpers for core CAP components."""

from abc import ABC
from typing import Any, ClassVar, Dict, Type


class RegisteredBase(ABC):
    """Base class mixin that provides decorator-based class registration."""

    _registry: ClassVar[Dict[str, Type["RegisteredBase"]]] = {}
    registry_key_method: ClassVar[str] = "registered_type"
    name: ClassVar[str] = ""

    @classmethod
    def register(cls):
        """Register a subclass using its registry key class method."""

        def decorator(subclass: Type["RegisteredBase"]) -> Type["RegisteredBase"]:
            key_method = getattr(subclass, cls.registry_key_method, None)
            if key_method is None or not callable(key_method):
                raise TypeError(
                    f"Subclass {subclass.__name__} must define "
                    f"{cls.registry_key_method}() class method"
                )
            if not getattr(subclass, "name", None):
                raise TypeError(
                    f"Subclass {subclass.__name__} must define name class attribute"
                )

            cls._registry[key_method()] = subclass
            return subclass

        return decorator

    @classmethod
    def get_registered_class(cls, registered_type: str) -> Type["RegisteredBase"] | None:
        """Get a registered class by its registry key."""
        return cls._registry.get(registered_type)

    @classmethod
    def get_registered_type(cls, registered_type: str) -> Type["RegisteredBase"] | None:
        """Get a registered class by its registry key."""
        return cls.get_registered_class(registered_type)

    @classmethod
    def registered_types(cls) -> list[str]:
        """Return all registered type keys for this base class."""
        return list(cls._registry.keys())

    @classmethod
    def get_registry(cls) -> Dict[str, Type["RegisteredBase"]]:
        """Return a shallow copy of the current registry."""
        return dict(cls._registry)

    @classmethod
    def registry(cls) -> Dict[str, Type["RegisteredBase"]]:
        """Return a shallow copy of the current registry."""
        return cls.get_registry()

    @classmethod
    def create(cls, registered_type: str, *args: Any, **kwargs: Any) -> "RegisteredBase":
        """Instantiate a registered class by type key."""
        subclass = cls.get_registered_class(registered_type)
        if subclass is None:
            raise KeyError(f"Unknown registered type: {registered_type}")
        return subclass(*args, **kwargs)
