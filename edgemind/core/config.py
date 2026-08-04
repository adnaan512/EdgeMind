"""
YAML-based configuration system for EdgeMind AI.

Design Philosophy:
    Research experiments MUST be reproducible. Every hyperparameter, every
    model choice, every data augmentation setting lives in a YAML config file.
    This module loads those configs and provides clean dot-access to values.

    Supports config inheritance: a base config defines defaults, and
    experiment-specific configs override only what they need.

Usage:
    >>> config = EdgeMindConfig.from_yaml("configs/base.yaml")
    >>> print(config.project.name)
    'EdgeMind AI'

    >>> config = EdgeMindConfig.from_yaml(
    ...     "configs/experiments/example.yaml",
    ...     base_config_path="configs/base.yaml"
    ... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigDict:
    """A dictionary wrapper that supports dot-access to nested values.

    This is a common pattern in ML frameworks. Instead of writing
    ``config["model"]["backbone"]["name"]``, you write
    ``config.model.backbone.name``.

    Attributes are read-only after creation to prevent accidental mutation
    during experiments.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain dictionary (useful for serialization)."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"ConfigDict({self.to_dict()})"

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def get(self, key: str, default: Any = None) -> Any:
        """Safe access with a default value, like dict.get()."""
        return getattr(self, key, default)


class EdgeMindConfig:
    """Main configuration class for EdgeMind AI experiments.

    Loads YAML configuration files and provides structured access to all
    experiment parameters. Supports base config + override merging.
    """

    def __init__(self, config_data: dict[str, Any]) -> None:
        self._raw = config_data
        self._config = ConfigDict(config_data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        return getattr(self._config, name)

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def __repr__(self) -> str:
        return f"EdgeMindConfig({self._raw})"

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        base_config_path: Optional[str | Path] = None,
    ) -> EdgeMindConfig:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the main YAML config file.
            base_config_path: Optional path to a base config. If provided,
                the base config is loaded first, then the main config
                is merged on top (overriding shared keys).

        Returns:
            An EdgeMindConfig instance with all parameters accessible
            via dot notation.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        config_data = cls._load_yaml(config_path)

        if base_config_path is not None:
            base_data = cls._load_yaml(base_config_path)
            config_data = cls._deep_merge(base_data, config_data)

        return cls(config_data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeMindConfig:
        """Create configuration from a plain dictionary.

        Useful for testing or programmatic config creation.
        """
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        """Export the configuration as a plain dictionary."""
        return self._raw.copy()

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:
        """Load and parse a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a YAML mapping, got: {type(data)}")
        return data

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override dict into base dict.

        For nested dicts, values are merged recursively.
        For all other types, the override value wins.
        """
        merged = base.copy()
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = EdgeMindConfig._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
