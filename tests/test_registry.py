"""Tests for the EdgeMind AI component registry."""

import pytest

from edgemind.core.registry import Registry


class TestRegistry:
    """Test the component registry pattern."""

    def test_register_and_get(self):
        """Should register a class and retrieve it by name."""
        reg = Registry("test")

        @reg.register("my_component")
        class MyComponent:
            pass

        assert reg.get("my_component") is MyComponent

    def test_auto_name(self):
        """When no name is given, should use lowercase class name."""
        reg = Registry("test")

        @reg.register()
        class MyModel:
            pass

        assert reg.get("mymodel") is MyModel

    def test_duplicate_registration(self):
        """Should raise KeyError on duplicate registration."""
        reg = Registry("test")

        @reg.register("dup")
        class First:
            pass

        with pytest.raises(KeyError, match="already registered"):
            @reg.register("dup")
            class Second:
                pass

    def test_get_missing(self):
        """Should raise KeyError with helpful message for missing keys."""
        reg = Registry("test")

        @reg.register("exists")
        class Exists:
            pass

        with pytest.raises(KeyError, match="not found"):
            reg.get("missing")

    def test_build(self):
        """Should instantiate a component from a config dict."""
        reg = Registry("test")

        @reg.register("adder")
        class Adder:
            def __init__(self, a: int, b: int):
                self.result = a + b

        instance = reg.build({"name": "adder", "a": 3, "b": 7})
        assert instance.result == 10

    def test_list_registered(self):
        """Should list all registered names in sorted order."""
        reg = Registry("test")

        @reg.register("beta")
        class Beta:
            pass

        @reg.register("alpha")
        class Alpha:
            pass

        assert reg.list_registered() == ["alpha", "beta"]

    def test_contains(self):
        """The `in` operator should check for registered names."""
        reg = Registry("test")

        @reg.register("present")
        class Present:
            pass

        assert "present" in reg
        assert "absent" not in reg

    def test_len(self):
        """len() should return the number of registered components."""
        reg = Registry("test")

        @reg.register("a")
        class A:
            pass

        @reg.register("b")
        class B:
            pass

        assert len(reg) == 2

    def test_register_function(self):
        """Should be able to register functions, not just classes."""
        reg = Registry("transforms")

        @reg.register("double")
        def double(x):
            return x * 2

        fn = reg.get("double")
        assert fn(5) == 10
