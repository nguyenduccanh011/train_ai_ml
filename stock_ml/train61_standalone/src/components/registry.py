from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any


class ComponentRegistryError(Exception):
    """Base error for component registry operations."""


class DuplicateComponentError(ComponentRegistryError):
    """Raised when a component name is already registered in a category."""


class ComponentNotFoundError(ComponentRegistryError):
    """Raised when a component name does not exist in a category."""


class ComponentRegistry:
    _instances: dict[str, dict[str, type[Any]]] = defaultdict(dict)
    _lock = RLock()

    @classmethod
    def register(
        cls,
        category: str,
        name: str,
        component_cls: type[Any],
        *,
        replace: bool = False,
    ) -> None:
        with cls._lock:
            bucket = cls._instances[category]
            if name in bucket and not replace:
                raise DuplicateComponentError(
                    f"Component '{name}' is already registered in category '{category}'."
                )
            bucket[name] = component_cls

    @classmethod
    def unregister(cls, category: str, name: str) -> None:
        with cls._lock:
            bucket = cls._instances.get(category, {})
            if name not in bucket:
                raise ComponentNotFoundError(
                    f"Component '{name}' is not registered in category '{category}'."
                )
            del bucket[name]

    @classmethod
    def get(cls, category: str, name: str) -> type[Any]:
        with cls._lock:
            bucket = cls._instances.get(category, {})
            component_cls = bucket.get(name)
            if component_cls is None:
                raise ComponentNotFoundError(
                    f"Unknown component '{name}' in category '{category}'."
                )
            return component_cls

    @classmethod
    def create(cls, category: str, name: str, **kwargs: Any) -> Any:
        component_cls = cls.get(category, name)
        return component_cls(**kwargs)

    @classmethod
    def list_components(cls, category: str | None = None) -> list[str]:
        with cls._lock:
            if category is not None:
                return sorted(cls._instances.get(category, {}).keys())
            names: list[str] = []
            for cat, bucket in sorted(cls._instances.items()):
                names.extend(f"{cat}.{key}" for key in sorted(bucket.keys()))
            return names

    @classmethod
    def categories(cls) -> list[str]:
        with cls._lock:
            return sorted(cls._instances.keys())

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._instances.clear()
