"""Tests for the EdgeMind AI device utilities."""

import torch

from edgemind.core.device import get_device, get_device_info, DeviceInfo


class TestGetDevice:
    """Test device detection logic."""

    def test_returns_torch_device(self):
        """Should return a torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_cpu_always_available(self):
        """CPU should always be selectable."""
        device = get_device(preferred="cpu")
        assert device.type == "cpu"

    def test_preferred_override(self):
        """When preferred is set, should use that device."""
        device = get_device(preferred="cpu")
        assert device.type == "cpu"

    def test_auto_detect(self):
        """Auto-detection should return a valid device type."""
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")


class TestGetDeviceInfo:
    """Test hardware info reporting."""

    def test_returns_device_info(self):
        """Should return a DeviceInfo dataclass."""
        info = get_device_info()
        assert isinstance(info, DeviceInfo)

    def test_has_pytorch_version(self):
        """DeviceInfo should include PyTorch version."""
        info = get_device_info()
        assert info.pytorch_version == torch.__version__

    def test_has_cpu_count(self):
        """DeviceInfo should report CPU thread count."""
        info = get_device_info()
        assert info.cpu_count > 0

    def test_str_output(self):
        """The string representation should be a formatted box."""
        info = get_device_info()
        text = str(info)
        assert "EdgeMind AI" in text
        assert "Device:" in text

    def test_cpu_info(self):
        """CPU device info should have device_name = 'CPU'."""
        info = get_device_info(preferred="cpu")
        assert info.device == "cpu"
        assert info.device_name == "CPU"
        assert info.total_memory_mb is None
