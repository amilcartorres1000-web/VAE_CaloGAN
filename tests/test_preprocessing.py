"""
Tests for preprocessing.
"""
import pytest
import numpy as np

from calogan_vae.data.preprocessing import Log1pNormStrategy, StandardizeStrategy


def test_log1p_norm_fit():
    """Test Log1pNorm fitting"""
    preprocessor = Log1pNormStrategy(percentile=99.0)
    data = np.random.exponential(scale=1.0, size=(100, 3, 12, 96))
    
    preprocessor.fit(data)
    
    assert preprocessor.scale_factor is not None
    assert preprocessor.scale_factor > 0
    assert preprocessor._fitted


def test_log1p_norm_preprocess():
    """Test Log1pNorm preprocessing"""
    preprocessor = Log1pNormStrategy(percentile=99.0)
    data = np.random.exponential(scale=1.0, size=(100, 3, 12, 96))
    
    preprocessor.fit(data)
    preprocessed = preprocessor.preprocess(data)
    
    assert preprocessed.shape == data.shape
    assert preprocessed.min() >= 0
    assert preprocessed.max() <= 1


def test_log1p_norm_inverse():
    """Test Log1pNorm inverse"""
    preprocessor = Log1pNormStrategy(percentile=99.0)
    data = np.random.exponential(scale=1.0, size=(100, 3, 12, 96))
    
    preprocessor.fit(data)
    preprocessed = preprocessor.preprocess(data)
    reconstructed = preprocessor.inverse(preprocessed)
    
    # Should be approximately equal
    np.testing.assert_allclose(data, reconstructed, rtol=0.2, atol=0.2)


def test_standardize():
    """Test Standardize preprocessing"""
    preprocessor = StandardizeStrategy()
    data = np.random.randn(100, 3, 12, 96).astype(np.float32)
    
    preprocessor.fit(data)
    preprocessed = preprocessor.preprocess(data)
    
    # Should have mean ~0 and std ~1
    assert np.abs(preprocessed.mean()) < 0.1
    assert np.abs(preprocessed.std() - 1.0) < 0.1


if __name__ == '__main__':
    pytest.main([__file__])