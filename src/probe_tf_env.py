import importlib.util
import platform
import sys

print("PY", platform.python_version())
found = importlib.util.find_spec("tensorflow") is not None
print("TF_FOUND", found)
if found:
    import tensorflow as tf
    print("TF_VER", tf.__version__)
