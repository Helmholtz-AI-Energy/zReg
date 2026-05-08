"""Root conftest - configure pytest path for src layout."""

import sys
from pathlib import Path

# Add src directory to Python path so imports work
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
