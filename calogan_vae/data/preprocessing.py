"""
Preprocessing strategies for calorimeter data.
All preprocessing must be invertible for validation.
"""
from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PreprocessingStrategy(ABC):
    """Base class for preprocessing strategies"""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Compute statistics from data"""
        pass
    
    @abstractmethod
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Transform data"""
        pass
    
    @abstractmethod
    def inverse(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform"""
        pass


class Log1pNormStrategy(PreprocessingStrategy):
    """
    Log1p followed by percentile normalization.
    
    This is what your original code did, but done correctly:
    1. Compute global statistics ONCE during fit()
    2. Apply consistently during preprocess()
    3. Invert correctly during inverse()
    """
    
    def __init__(self, percentile: float = 99.0):
        self.percentile = percentile
        self.scale_factor = None
        self._fitted = False
    
    def fit(self, data: np.ndarray) -> None:
        """
        Compute global statistics from data.
        
        Args:
            data: Array of shape (N, C, H, W) or similar
        """
        # Get all non-zero values
        non_zero = data[data > 1e-6]
        
        if len(non_zero) == 0:
            logger.warning("No non-zero values found, using scale_factor=1.0")
            self.scale_factor = 1.0
        else:
            # Apply log1p first, then compute percentile
            log_non_zero = np.log1p(non_zero)
            self.scale_factor = np.percentile(log_non_zero, self.percentile)
            
            # Ensure it's not too small
            self.scale_factor = max(self.scale_factor, 1.0)
        
        self._fitted = True
        logger.info(f"Log1pNorm fitted: scale_factor={self.scale_factor:.4f}")
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Transform: log1p -> normalize -> clip
        
        Args:
            data: Raw calorimeter data
            
        Returns:
            Preprocessed data in [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before preprocess()")
        
        # Log transform
        data = np.log1p(data)
        
        # Normalize by global scale
        data = data / self.scale_factor
        
        # Clip to valid range
        data = np.clip(data, 0, 1)
        
        return data.astype(np.float32)
    
    def inverse(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform: denormalize -> expm1
        
        Args:
            data: Preprocessed data in [0, 1]
            
        Returns:
            Original scale data
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before inverse()")
        
        # ADD DEBUGGING
        print(f"[DEBUG] Input to inverse:")
        print(f"  Mean: {data.mean():.6f}")
        print(f"  Std: {data.std():.6f}")
        print(f"  Min: {data.min():.6f}")
        print(f"  Max: {data.max():.6f}")
        
        # Denormalize
        data = data * self.scale_factor
        
        print(f"[DEBUG] After denormalization:")
        print(f"  Mean: {data.mean():.6f}")
        print(f"  scale_factor: {self.scale_factor:.6f}")

        # Inverse log1p
        data = np.expm1(data)

        print(f"[DEBUG] After expm1:")
        print(f"  Mean: {data.mean():.6f}") 
        
        # Remove negative values (numerical errors)
        data = np.maximum(data, 0)
        
        return data.astype(np.float32)


class StandardizeStrategy(PreprocessingStrategy):
    """
    Standard z-score normalization (mean=0, std=1).
    Alternative to Log1p if you want to try it.
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self._fitted = False
    
    def fit(self, data: np.ndarray) -> None:
        """Compute mean and std"""
        self.mean = data.mean()
        self.std = data.std()
        
        # Avoid division by zero
        if self.std < 1e-6:
            self.std = 1.0
        
        self._fitted = True
        logger.info(f"Standardize fitted: mean={self.mean:.4f}, std={self.std:.4f}")
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Standardize"""
        if not self._fitted:
            raise RuntimeError("Must call fit() before preprocess()")
        
        return ((data - self.mean) / self.std).astype(np.float32)
    
    def inverse(self, data: np.ndarray) -> np.ndarray:
        """Destandardize"""
        if not self._fitted:
            raise RuntimeError("Must call fit() before inverse()")
        
        return (data * self.std + self.mean).astype(np.float32)


def get_preprocessor(name: str, **kwargs) -> PreprocessingStrategy:
    """
    Factory function to create preprocessors.
    
    Args:
        name: Preprocessor name ('log1p_norm', 'standardize')
        **kwargs: Arguments to pass to preprocessor
        
    Returns:
        Preprocessor instance
    """
    if name == 'log1p_norm':
        return Log1pNormStrategy(**kwargs)
    elif name == 'standardize':
        return StandardizeStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown preprocessor: {name}")