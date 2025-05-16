"""
GlucoseTransformer Package

This package provides models and utilities for predicting glucose levels 
using transformer-based deep learning architectures.
"""

__version__ = '0.1.0'

# Import key classes to expose at the package level
from .models import TransformerEncoder, TransformerEncoder_version2, PositionalEncoding
from .data import TimeSeriesDataset, load_ohio_series_train, split_into_continuous_series
from .train import train_model, evaluate_model
from .utils import evaluate_and_save_metrics, save_model, load_model

# Define what symbols are exported when using from src import *
__all__ = [
    # Models
    'TransformerEncoder',
    'TransformerEncoder_version2',
    'PositionalEncoding',
    
    # Data processing
    'TimeSeriesDataset',
    'load_ohio_series_train',
    'split_into_continuous_series',
    'create_train_val_datasets',
    
    # Training
    'train_model',
    'evaluate_model',
    
    # Utilities
    'evaluate_and_save_metrics',
    'evaluate_and_save_metrics_population',
    'save_model',
    'load_model',
    'load_model_population'
]

# Package metadata
PACKAGE_METADATA = {
    'name': 'GlucoseTransformer',
    'description': 'Transformer-based models for glucose prediction',
    'requires': [
        'torch>=1.10.0',
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.24.0'
    ]
}

# Check for required dependencies
def _check_dependencies():
    """Verify that required dependencies are available."""
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
        
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
        
    try:
        import matplotlib
    except ImportError:
        missing_deps.append('matplotlib')
        
    try:
        import sklearn
    except ImportError:
        missing_deps.append('scikit-learn')
    
    if missing_deps:
        print(f"Warning: Missing required dependencies: {', '.join(missing_deps)}")
        print("Please install these packages to ensure full functionality.")

# Run dependency check when package is imported
_check_dependencies()

print(f"GlucoseTransformer v{__version__} loaded successfully.")