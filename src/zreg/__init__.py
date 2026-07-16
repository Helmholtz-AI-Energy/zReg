import os
import logging
from importlib.metadata import PackageNotFoundError, version

from .setup_log import setup_logger

_log_level_str = os.environ.get("ZREG_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)
setup_logger(_log_level)


def set_log_level(level: int | str) -> None:
    """Set the log level for the zreg logger.

    Parameters
    ----------
    level : int or str
        Logging level. Accepts both string (e.g., "WARNING") and
        int (e.g., logging.WARNING) values.
    """
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

from . import dataset as dataset  # noqa: E402
from . import transforms as transforms  # noqa: E402
from . import downsampling as downsampling  # noqa: E402
from . import utils as utils
from . import config as config
from . import cpd as cpd
from . import distances as distances
from . import pairwise_distance_matrix as pairwise_distance_matrix
from . import dtw as dtw
from . import color_transfer as color_transfer
from . import preprocessing as preprocessing
