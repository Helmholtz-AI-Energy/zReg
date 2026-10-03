import os
import logging
from importlib.metadata import PackageNotFoundError, version

# scipy.io must be imported before torch on macOS ARM to avoid duplicate libomp
# initialisation (SIGABRT / SIGSEGV).  Importing `from .utils.setup_log import
# setup_logger` triggers zreg.utils.__init__ which starts with `import torch`;
# putting scipy first here ensures the conda libomp is already loaded when torch
# initialises, so torch reuses it rather than opening a second copy.
import scipy.io as _sio  # noqa: F401 — side-effect import for libomp ordering
del _sio

from .utils.setup_log import setup_logger

_log_level_str = os.environ.get("ZREG_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)
setup_logger(_log_level)


def set_log_level(level: int | str) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logging.getLogger("zreg").setLevel(level)


try:
    dist_name = "zReg"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from . import core as core  # noqa: E402
from . import preprocessing as preprocessing  # noqa: E402
from . import distance_metrics as distance_metrics  # noqa: E402
from . import algorithms as algorithms  # noqa: E402
from . import evaluation as evaluation  # noqa: E402
from . import data_generation as data_generation  # noqa: E402
from . import utils as utils  # noqa: E402
from . import config as config  # noqa: E402
from . import label_transfer as label_transfer  # noqa: E402

# Backward-compatibility shim: ``zreg.dtw`` → ``zreg.algorithms.dtw``.  It is a
# real module alias (registered in sys.modules), not only an attribute, so that
# ``from zreg.dtw import X`` and ``importlib.import_module("zreg.dtw")`` work.
# The submodules are aliased too (WR-05): otherwise ``import zreg.dtw.core``
# would re-execute ``core.py`` under a new name (failing on its relative
# imports, or creating duplicate classes that break ``isinstance``).
# No other dropped top-level module is aliased.
import importlib as _importlib  # noqa: E402
import sys as _sys  # noqa: E402

dtw = algorithms.dtw
_sys.modules[__name__ + ".dtw"] = algorithms.dtw
for _sub in ("core", "result", "constraints"):
    _sys.modules[f"{__name__}.dtw.{_sub}"] = _importlib.import_module(
        f".algorithms.dtw.{_sub}", __name__
    )
del _sys, _importlib, _sub
# zreg.models is intentionally NOT eagerly imported here.  Its transitive import
# of torch_geometric initialises a second libomp copy on macOS ARM and causes a
# SIGABRT / SIGSEGV when open3d is later loaded by algorithms.icp.  Callers that
# need the models sub-package should import it explicitly: `from zreg import models`.
# This matches the original zreg/__init__.py behaviour (models was not auto-imported).
