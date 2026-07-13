"""zreg.models — hand-rolled point-cloud deep learning models (PointNet++, eGNN).

Exports the shared, non-differentiable geometry ops (``_ops.py``) consumed by
both model architectures — ``farthest_point_sample``, ``ball_query``,
``build_radius_graph`` — plus the two joint-cloud label-transfer models,
``PointNet2LabelTransfer`` (``.pointnet2``) and ``EGNNLabelTransfer``
(``.egnn``).

Import order note (47-RESEARCH.md Pitfall 1): importing this package
transitively imports ``torch_geometric`` via ``.egnn``, which crashes with a
libomp SIGABRT on macOS ARM if no ``zreg``/``scipy``-importing module has been
imported yet in the process. ``.pointnet2``/``.egnn`` each already guard this
themselves (``from zreg.dataset import zRegPointCloud`` as their first
import), and this package lives under ``zreg`` itself, so any caller that
reaches ``zreg.models`` has necessarily already imported ``zreg`` — no
additional guard is needed here.
"""

from zreg.models._ops import ball_query, build_radius_graph, farthest_point_sample
from zreg.models.egnn import EGNNLabelTransfer
from zreg.models.pointnet2 import PointNet2LabelTransfer

__all__ = [
    "farthest_point_sample",
    "ball_query",
    "build_radius_graph",
    "PointNet2LabelTransfer",
    "EGNNLabelTransfer",
]
