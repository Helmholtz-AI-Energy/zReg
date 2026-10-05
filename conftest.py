"""Root conftest - configure pytest path for src layout."""

import os
import sys
from pathlib import Path

# On macOS ARM, importing both torch and torch_geometric (or open3d) initialises two
# separate copies of libomp and triggers SIGABRT (exit 134).  Setting this env var
# before *any* library import suppresses the fatal exit so the process continues.
# This is the same workaround applied in the main-branch conftest and is safe for CI.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add src directory to Python path so imports work
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
