"""Experiment run metadata writer for the zReg evaluation framework.

This package provides the ``log_run()`` function for recording experiment run
metadata to local JSON and CSV files using stdlib only.  Each call to
``log_run()`` produces two output files: ``{run_id}.json`` and
``{run_id}.csv``, written to ``experiments/runs/`` by default.
``export_trajectory`` writes point-per-row CSV trajectory files and
accompanying metadata JSON per EXT-01.
"""

from .tracking import log_run
from .trajectory import export_trajectory

__all__ = ["log_run", "export_trajectory"]
