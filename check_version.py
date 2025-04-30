import sys
import torch
import tensorflow as tf
import sklearn
import scipy
import numpy
import pandas

def print_versions():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"SciPy version: {scipy.__version__}")
    print(f"NumPy version: {numpy.__version__}")
    print(f"Pandas version: {pandas.__version__}")

if __name__ == "__main__":
    print_versions()
