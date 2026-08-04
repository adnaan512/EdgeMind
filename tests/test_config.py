"""Tests for the EdgeMind AI configuration system."""

import tempfile
from pathlib import Path

import pytest

from edgemind.core.config import EdgeMindConfig, ConfigDict


class TestConfigDict:
    """Test the ConfigDict dot-access wrapper."""

    def test_dot_access(self):
        """Nested dicts should be accessible via dot notation."""
        data = {"model": {"name": "resnet18", "num_classes": 10}}
        cd = ConfigDict(data)
        assert cd.model.name == "resnet18"
        assert cd.model.num_classes == 10

    def test_to_dict(self):
        """ConfigDict should convert back to a plain dict."""
        data = {"a": 1, "b": {"c": 2}}
        cd = ConfigDict(data)
        assert cd.to_dict() == data

    def test_contains(self):
        """The `in` operator should check for top-level keys."""
        cd = ConfigDict({"x": 1, "y": 2})
        assert "x" in cd
        assert "z" not in cd

    def test_get_with_default(self):
        """get() should return a default for missing keys."""
        cd = ConfigDict({"a": 1})
        assert cd.get("a") == 1
        assert cd.get("missing", 42) == 42


class TestEdgeMindConfig:
    """Test YAML loading and config merging."""

    def _write_yaml(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def test_from_yaml(self, tmp_path):
        """Should load a simple YAML file."""
        config_file = tmp_path / "test.yaml"
        self._write_yaml(config_file, "project:\n  name: TestProject\nseed: 123\n")

        config = EdgeMindConfig.from_yaml(config_file)
        assert config.project.name == "TestProject"
        assert config.seed == 123

    def test_from_dict(self):
        """Should create config from a plain dictionary."""
        config = EdgeMindConfig.from_dict({"model": {"name": "test"}})
        assert config.model.name == "test"

    def test_deep_merge(self, tmp_path):
        """Experiment config should override base config values."""
        base_file = tmp_path / "base.yaml"
        self._write_yaml(base_file, (
            "model:\n"
            "  name: mobilenet\n"
            "  num_classes: 10\n"
            "training:\n"
            "  epochs: 100\n"
            "  lr: 0.01\n"
        ))

        exp_file = tmp_path / "exp.yaml"
        self._write_yaml(exp_file, (
            "training:\n"
            "  epochs: 5\n"
        ))

        config = EdgeMindConfig.from_yaml(exp_file, base_config_path=base_file)

        # Overridden value
        assert config.training.epochs == 5
        # Inherited values
        assert config.model.name == "mobilenet"
        assert config.training.lr == 0.01

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            EdgeMindConfig.from_yaml("nonexistent.yaml")

    def test_to_dict(self):
        """Should export config back to dict."""
        data = {"a": 1, "b": {"c": 2}}
        config = EdgeMindConfig.from_dict(data)
        assert config.to_dict() == data

    def test_empty_yaml(self, tmp_path):
        """Should handle empty YAML files gracefully."""
        config_file = tmp_path / "empty.yaml"
        self._write_yaml(config_file, "")

        config = EdgeMindConfig.from_yaml(config_file)
        assert config.to_dict() == {}
