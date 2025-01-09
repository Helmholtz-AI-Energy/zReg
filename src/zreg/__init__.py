import sys
import logging
from .setup_log import setup_logger

setup_logger(logging.INFO)  # TODO: add a basic flag to overwrite this?

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
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
from . import cpd as cpd
from . import distances as distances
from . import dtw as dtw
# from . import mpi_tools as mpi_tools
