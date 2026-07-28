# dataset must be imported before types on macOS ARM: dataset.py starts with
# `import scipy.io` which initialises libomp via scipy, ensuring that torch
# (imported by types.py immediately after) reuses the already-loaded libomp
# rather than loading a second copy (which causes SIGABRT / SIGSEGV with open3d).
from . import dataset as dataset
from . import types as types
from . import transforms as transforms
