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

# Backward-compatibility shims: expose sub-modules at the old top-level paths so
# that existing code / tests using e.g. ``zreg.dtw`` still works.
dtw = algorithms.dtw  # zreg.dtw → zreg.algorithms.dtw
# zreg.models is intentionally NOT eagerly imported here.  Its transitive import
# of torch_geometric initialises a second libomp copy on macOS ARM and causes a
# SIGABRT / SIGSEGV when open3d is later loaded by algorithms.icp.  Callers that
# need the models sub-package should import it explicitly: `from zreg import models`.
# This matches the original zreg/__init__.py behaviour (models was not auto-imported).
