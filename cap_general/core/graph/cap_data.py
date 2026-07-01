"""CapData: typed key-value container for data flowing between DAG nodes."""

from __future__ import annotations

from typing import Any


class CapData:
    """Key-value store for inputs and outputs between CapGraph nodes."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.values: dict[str, Any] = {}

    def __str__(self) -> str:
        label = self.name or "cap_data"
        if not self.values:
            return f"{label}(empty)"
        items = [f"{k}={v!r}" for k, v in list(self.values.items())[:8]]
        return f"{label}({', '.join(items)})"

    @property
    def datas(self) -> dict[str, Any]:
        return self.values

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> CapData:
        self.values[key] = value
        return self

    def update(self, data: dict[str, Any]) -> CapData:
        self.values.update(data)
        return self

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str | None = None) -> CapData:
        obj = cls(name=name)
        obj.values = dict(data)
        return obj
