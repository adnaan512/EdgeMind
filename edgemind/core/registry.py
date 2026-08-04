"""
Component Registry for EdgeMind AI.

Design Philosophy:
    Research frameworks need a way to instantiate components from config
    strings. Instead of writing giant if/elif chains, we use a registry
    that maps string names to Python classes.

    This is the exact same pattern used by:
    - MMDetection (OpenMMLab)
    - Detectron2 (Facebook AI Research)
    - timm (Ross Wightman)

Usage:
    >>> MODELS = Registry("models")

    >>> @MODELS.register("my_model")
    ... class MyModel:
    ...     def __init__(self, num_classes: int = 10):
    ...         self.num_classes = num_classes

    >>> model_class = MODELS.get("my_model")
    >>> model = model_class(num_classes=5)

    >>> # Or build directly from a config dict:
    >>> model = MODELS.build({"name": "my_model", "num_classes": 5})
"""

from __future__ import annotations

from typing import Any, Callable, Optional


class Registry:
    """A named registry that maps string keys to Python classes or functions.

    Each major component type (models, datasets, optimizers, etc.) gets its
    own Registry instance. Components register themselves via the
    ``@registry.register()`` decorator.

    Args:
        name: A human-readable name for this registry (e.g., "models").
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type | Callable] = {}

    def register(
        self,
        name: Optional[str] = None,
    ) -> Callable:
        """Decorator to register a class or function under a given name.

        Args:
            name: The key to register under. If None, uses the class/function
                name in lowercase.

        Returns:
            The original class/function, unmodified.

        Example:
            >>> MODELS = Registry("models")
            >>> @MODELS.register("resnet18")
            ... class ResNet18Wrapper:
            ...     pass
        """
        def decorator(cls_or_fn: type | Callable) -> type | Callable:
            key = name if name is not None else cls_or_fn.__name__.lower()

            if key in self._registry:
                raise KeyError(
                    f"'{key}' is already registered in the '{self.name}' registry. "
                    f"Existing: {self._registry[key]}, New: {cls_or_fn}"
                )

            self._registry[key] = cls_or_fn
            return cls_or_fn

        return decorator

    def get(self, name: str) -> type | Callable:
        """Retrieve a registered component by name.

        Args:
            name: The registered key.

        Returns:
            The registered class or function.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in the '{self.name}' registry. "
                f"Available: [{available}]"
            )
        return self._registry[name]

    def build(self, config: dict[str, Any]) -> Any:
        """Instantiate a registered component from a config dictionary.

        The config must contain a ``"name"`` key. All other keys are passed
        as keyword arguments to the constructor.

        Args:
            config: A dict with ``"name"`` and optional constructor kwargs.

        Returns:
            An instance of the registered component.

        Example:
            >>> model = MODELS.build({"name": "resnet18", "num_classes": 10})
        """
        config = config.copy()
        name = config.pop("name")
        cls_or_fn = self.get(name)
        return cls_or_fn(**config)

    def list_registered(self) -> list[str]:
        """Return a sorted list of all registered component names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        items = ", ".join(self.list_registered())
        return f"Registry(name='{self.name}', items=[{items}])"
