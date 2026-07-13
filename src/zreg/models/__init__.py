"""zreg.models — hand-rolled point-cloud deep learning models (PointNet++, eGNN).

Currently exports the shared, non-differentiable geometry ops (``_ops.py``)
consumed by both model architectures: ``farthest_point_sample``, ``ball_query``,
and ``build_radius_graph``. Model classes (``PointNet2LabelTransfer``,
``EGNNLabelTransfer``) are added to this package in a later plan (47-04).
"""

from zreg.models._ops import ball_query, build_radius_graph, farthest_point_sample

__all__ = ["farthest_point_sample", "ball_query", "build_radius_graph"]
