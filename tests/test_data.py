"""Tests for datasets and dataloader factory."""

from unittest import mock

import pytest
from torch.utils.data import DataLoader

from edgemind.core.config import EdgeMindConfig
from edgemind.data import DATASETS
from edgemind.data.dataloader import build_dataloaders


class DummyDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __len__(self):
        return 10
    def __getitem__(self, idx):
        return None, 0


def test_registry_contains_datasets():
    """Ensure datasets are registered properly."""
    assert "cifar10" in DATASETS
    assert "custom_folder" in DATASETS


@mock.patch("edgemind.data.dataloader.DATASETS.build")
def test_build_dataloaders(mock_build):
    """Test that the dataloader factory passes correct config to registry."""
    # Setup mock
    mock_build.return_value = DummyDataset()
    
    # Create a config
    config = EdgeMindConfig.from_dict({
        "data": {
            "dataset": "test_dataset",
            "batch_size": 2, # Note: batch_size usually in training, testing robust handling
            "num_workers": 0
        },
        "training": {
            "batch_size": 4
        }
    })
    
    # Build
    train_loader, val_loader = build_dataloaders(config)
    
    # Verify mock was called twice (train and val)
    assert mock_build.call_count == 2
    
    # Verify DataLoaders
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 4
    
    # Train is shuffled, drop_last=True
    # Val is not shuffled, drop_last=False
