"""
Tests for dataset module.
"""
import pytest
import numpy as np
import torch
import tempfile
import h5py
from pathlib import Path

from calogan_vae.data import CaloDataset
from calogan_vae.data.preprocessing import Log1pNormStrategy


def create_dummy_h5(path: Path, n_events: int = 100):
    """Create a dummy HDF5 file for testing"""
    with h5py.File(path, 'w') as f:
        # Create 3 layers with random data
        for i in range(3):
            data = np.random.exponential(scale=1.0, size=(n_events, 12, 96))
            f.create_dataset(f'layer_{i}', data=data.astype(np.float32))


def test_dataset_creation():
    """Test dataset can be created"""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / 'test.h5'
        create_dummy_h5(h5_path, n_events=50)
        
        dataset = CaloDataset(
            h5_path=str(h5_path),
            layer_keys=['layer_0', 'layer_1', 'layer_2'],
            max_events=50,
            loading_mode='memory'
        )
        
        assert len(dataset) == 50
        assert dataset.n_events == 50


def test_dataset_getitem():
    """Test getting items from dataset"""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / 'test.h5'
        create_dummy_h5(h5_path, n_events=50)
        
        dataset = CaloDataset(
            h5_path=str(h5_path),
            layer_keys=['layer_0', 'layer_1', 'layer_2'],
            max_events=50,
            loading_mode='memory'
        )
        
        # Get item
        sample = dataset[0]
        
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (3, 12, 96)
        assert sample.dtype == torch.float32


def test_preprocessing_invertibility():
    """Test that preprocessing is invertible"""
    preprocessor = Log1pNormStrategy(percentile=99.0)
    
    # Create dummy data
    data = np.random.exponential(scale=1.0, size=(100, 3, 12, 96)).astype(np.float32)
    
    # Fit
    preprocessor.fit(data)
    
    # Preprocess
    data_preprocessed = preprocessor.preprocess(data)
    
    # Inverse
    data_reconstructed = preprocessor.inverse(data_preprocessed)
    
    # Check similarity (won't be exact due to clipping)
    np.testing.assert_allclose(data, data_reconstructed, rtol=0.1, atol=0.1)


def test_dataset_loading_modes():
    """Test different loading modes"""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / 'test.h5'
        create_dummy_h5(h5_path, n_events=50)
        
        for mode in ['memory', 'disk']:
            dataset = CaloDataset(
                h5_path=str(h5_path),
                layer_keys=['layer_0', 'layer_1', 'layer_2'],
                max_events=50,
                loading_mode=mode
            )
            
            sample = dataset[0]
            assert sample.shape == (3, 12, 96)


if __name__ == '__main__':
    pytest.main([__file__])