"""
Setup script for calogan_vae package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text() if readme_path.exists() else ''

setup(
    name='calogan_vae',
    version='0.1.0',
    description='Variational Autoencoder for CaloGAN shower simulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/calogan_vae',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'h5py>=3.8.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)