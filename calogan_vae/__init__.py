"""
CaloGAN VAE Package

A modular implementation of Variational Autoencoders for calorimeter shower simulation.
"""

__version__ = '0.1.0'

from . import config
from . import data
from . import models
from . import training
from . import visualization
from . import utils

__all__ = [
    'config',
    'data',
    'models',
    'training',
    'visualization',
    'utils'
]