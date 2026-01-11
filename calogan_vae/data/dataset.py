"""
Calorimeter dataset with efficient loading strategies.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Literal, Callable
import logging
import math

logger = logging.getLogger(__name__)


class CaloDataset(Dataset):
    """
    Calorimeter dataset with smart loading strategies.
    
    Strategies:
    - 'memory': Load entire dataset into RAM (fast, memory-intensive)
    - 'cache': LRU cache for frequently accessed samples (balanced)
    - 'disk': Read from disk each time (slow, memory-efficient)
    - 'auto': Automatically choose based on dataset size
    
    Args:
        h5_path: Path to HDF5 file
        layer_keys: Keys for layer datasets in HDF5
        max_events: Maximum number of events to use
        loading_mode: Loading strategy
        transform: Optional transform function
        preprocessor: Optional preprocessing strategy
    """
    
    def __init__(
        self,
        h5_path: str,
        layer_keys: List[str] = None,
        max_events: Optional[int] = None,
        loading_mode: Literal['auto', 'memory', 'cache', 'disk'] = 'auto',
        transform: Optional[Callable] = None,
        preprocessor: Optional[Callable] = None
    ):
        self.h5_path = Path(h5_path)
        self.transform = transform
        self.preprocessor = preprocessor
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        
        # Determine layer keys
        with h5py.File(self.h5_path, 'r') as f:
            if layer_keys is None:
                # Auto-detect layer keys
                all_keys = list(f.keys())
                layer_keys = [k for k in all_keys if 'layer' in k.lower()]
                if len(layer_keys) == 0:
                    layer_keys = [k for k in ['layer_0', 'layer_1', 'layer_2'] 
                                 if k in f.keys()]
            
            self.layer_keys = layer_keys[:3]  # Use first 3 layers
            
            # Determine dataset size
            n_total = f[self.layer_keys[0]].shape[0]
            self.n_events = n_total if max_events is None else min(n_total, max_events)
            
            # Estimate memory usage
            sample_shape = f[self.layer_keys[0]].shape[1:]
            bytes_per_sample = np.prod(sample_shape) * 4 * len(self.layer_keys)  # float32
            total_bytes = bytes_per_sample * self.n_events
            total_gb = total_bytes / (1024 ** 3)
        
        logger.info(f"Dataset: {self.n_events} events from {self.h5_path.name}")
        logger.info(f"Layers: {self.layer_keys}")
        logger.info(f"Estimated size: {total_gb:.2f} GB")
        
        # Choose loading strategy
        if loading_mode == 'auto':
            if total_gb < 5.0:
                loading_mode = 'memory'
            elif total_gb < 20.0:
                loading_mode = 'cache'
            else:
                loading_mode = 'disk'
        
        self.loading_mode = loading_mode
        logger.info(f"Loading mode: {self.loading_mode}")
        
        # Load data according to strategy
        if self.loading_mode == 'memory':
            self.data = self._load_to_memory()
            self._get_item_fn = self._get_from_memory
        elif self.loading_mode == 'cache':
            from functools import lru_cache
            self.cache_size = min(1000, self.n_events)
            self._get_item_fn = lru_cache(maxsize=self.cache_size)(self._get_from_disk)
        else:  # disk
            self._get_item_fn = self._get_from_disk
    
    def _load_to_memory(self) -> np.ndarray:
        """Load entire dataset into RAM"""
        logger.info("Loading dataset into memory...")
        
        with h5py.File(self.h5_path, 'r') as f:
            layers = []
            for key in self.layer_keys:
                layer_data = np.array(f[key][:self.n_events], dtype=np.float32)
                layers.append(layer_data)
        
        # Stack layers: (N, H, W, ...) -> (N, 3, H, W)
        data = self._process_layers(layers)
        
        logger.info(f"Loaded {data.shape[0]} events into memory")
        return data
    
    def _get_from_memory(self, idx: int) -> np.ndarray:
        """Get sample from memory"""
        return self.data[idx]
    
    def _get_from_disk(self, idx: int) -> np.ndarray:
        """Get sample from disk"""
        with h5py.File(self.h5_path, 'r') as f:
            layers = [np.array(f[key][idx], dtype=np.float32) 
                     for key in self.layer_keys]
        
        return self._process_single_event(layers)
    
    def _process_layers(self, layers: List[np.ndarray]) -> np.ndarray:
        """
        Process multiple layers into fixed (N, 3, 12, 96) shape.
        
        Args:
            layers: List of 3 layer arrays, each of shape (N, H, W) or (N, H, W, C)
            
        Returns:
            Array of shape (N, 3, 12, 96)
        """
        n_events = layers[0].shape[0]
        H, W = 12, 96
        canvas = np.zeros((n_events, 3, H, W), dtype=np.float32)
        
        for i, layer in enumerate(layers):
            # Handle different shapes
            if layer.ndim == 4 and layer.shape[-1] == 1:
                layer = layer[..., 0]  # (N, H, W, 1) -> (N, H, W)
            
            if layer.ndim == 3:
                h, w = layer.shape[1], layer.shape[2]
            else:
                # Flatten and reshape
                layer = layer.reshape(n_events, -1)
                h = int(math.sqrt(layer.shape[1]))
                w = layer.shape[1] // h
                layer = layer.reshape(n_events, h, w)
            
            # Copy to canvas (crop if needed)
            h_copy = min(h, H)
            w_copy = min(w, W)
            canvas[:, i, :h_copy, :w_copy] = layer[:, :h_copy, :w_copy]
        
        return canvas
    
    def _process_single_event(self, layers: List[np.ndarray]) -> np.ndarray:
        """
        Process single event layers into (3, 12, 96) shape.
        
        Args:
            layers: List of 3 layer arrays
            
        Returns:
            Array of shape (3, 12, 96)
        """
        H, W = 12, 96
        canvas = np.zeros((3, H, W), dtype=np.float32)
        
        for i, layer in enumerate(layers):
            # Handle different shapes
            if layer.ndim == 3 and layer.shape[0] == 1:
                layer = layer[0]
            
            if layer.ndim != 2:
                # Try to reshape
                try:
                    size = int(math.sqrt(layer.size))
                    layer = layer.reshape(size, -1)
                except:
                    layer = layer.reshape(-1, 1)
            
            h, w = layer.shape
            h_copy = min(h, H)
            w_copy = min(w, W)
            canvas[i, :h_copy, :w_copy] = layer[:h_copy, :w_copy]
        
        return canvas
    
    def __len__(self) -> int:
        return self.n_events
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample.
        
        Returns:
            Tensor of shape (3, 12, 96)
        """
        # Get raw data
        data = self._get_item_fn(idx)
        
        # Apply preprocessing
        if self.preprocessor is not None:
            data = self.preprocessor.preprocess(data)
        
        # Apply transform
        if self.transform is not None:
            data = self.transform(data)
        
        return torch.from_numpy(data).float()