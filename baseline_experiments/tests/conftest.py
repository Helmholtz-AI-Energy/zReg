"""conftest.py for baseline_experiments tests.

Inserts baseline_experiments/scripts onto sys.path so tests can import
aggregate_cost and extract_calibration directly (without subprocess).
"""

import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
