"""Unmocked direction-contract tests for ``cpd_weighted`` label transfer (Phase 59 NUM-04).

Review HIGH 1 (59-REVIEWS.md): the Kobitski -> Shah ``cpd_weighted`` experiment
must transfer Shah's germ-layer labels (the target, provider) onto the aligned
Kobitski source (receiver) when ``label_source == "target"``.

``AlignmentStage`` stores, per target key ``tk``, the E-step of
``RigidCPD.expectation_step(t_source=aligned_source, target=target)``, whose
``pmat`` is shaped ``(n_aligned_source, n_target)``.  In the default direction
the stage transposes it (receiver = target); in the target direction it must be
used as stored (receiver = aligned source).  Unequal point counts and disjoint
label classes make an orientation error observable.  No mocks or patches: the
posteriors come from the real ``expectation_step`` and the runner-level tests
drive the real ``AlignmentStage`` / ``LabelTransferStage`` / ``MetricsEngine``.
"""

import time
from pathlib import Path

# zreg.* before torch before eval.* (macOS-ARM libomp SIGABRT rule)
from zreg import utils
from zreg.algorithms.cpd.rigid import RigidCPD
from zreg.core.dataset import zRegPointCloud

import matplotlib

matplotlib.use("Agg")

import pytest
import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.runners import EvaluationRunner
from eval.stages.label_transfer import LabelTransferStage
from eval.types import AlignResult

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_YAML = REPO_ROOT / "configs" / "experiments" / "stage2_label_transfer" / "cpd_weighted" / "real.yaml"

CENTRES = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
PROVIDER_CLASSES = (10, 11, 12)
PROVIDER_PER_CLUSTER = 14  # 42 points per provider frame
RECEIVER_PER_CLUSTER = 9  # 27 points per receiver frame

LT_PARAMS = {
    "k_neighbours": 5,
    "dist_metric": "euclidean",
    "smoothing": 0.0,
    "threshold": 0.5,
    "method": "cpd_weighted",
}
RIGID_ALIGN_PARAMS = {
    "window_size": 10,
    "step": 1,
    "cpd_penalty": "rigid",
    "dtw_dist_fn": "euclidean",
    "n_breakpoints": 5,
    "alignment_method": "cpd",
}


def _clustered(per_cluster, seed, labelled):
    """Points around the three CENTRES; returns (cloud, cluster index per point)."""
    gen = torch.Generator().manual_seed(seed)
    cluster = torch.arange(3).repeat_interleave(per_cluster)
    pos = CENTRES[cluster] + 0.5 * torch.randn(cluster.numel(), 3, generator=gen)
    fields = {"pos": pos}
    if labelled:
        fields["label"] = torch.tensor(PROVIDER_CLASSES)[cluster]
    return zRegPointCloud(**fields), cluster


def _estep(aligned_source_pos, target_pos, sigma2=None):
    """Exactly what AlignmentStage stores: t_source = aligned source, target = target."""
    if sigma2 is None:
        sigma2 = utils.squared_kernel_sum(aligned_source_pos, target_pos)
    cpd = RigidCPD(source=aligned_source_pos, use_color=False)
    return cpd.expectation_step(
        t_source=aligned_source_pos, target=target_pos, sigma2=sigma2, sigma2_c=0.0, w=0.0
    )


def _align_result(aligned_cloud, estep_results):
    return AlignResult(
        aligned_cloud=aligned_cloud,
        warp_path=[(0, 0)],
        dtw_distance=0.0,
        n_changepoints=0,
        params_used={},
        estep_results=estep_results,
    )


def _stage_cfg(tmp_path, label_source):
    kwargs = dict(
        data_path=str(tmp_path / "unused.tracklets"),
        label_transfer_method="cpd_weighted",
        label_source=label_source,
        output_dir=str(tmp_path / "out"),
        save_plots=False,
    )
    if label_source == "target":
        kwargs.update(pipeline_mode="paired", target_data_path=str(tmp_path / "t.csv"))
    return EvalConfig(**kwargs)


def _cluster_accuracy(labels, cluster):
    expected = torch.tensor(PROVIDER_CLASSES)[cluster]
    return (labels == expected).float().mean().item()


def _values(tensor):
    return set(int(v) for v in tensor.unique().tolist())


# ---------------------------------------------------------------------------
# Stage level (real CPD posterior)
# ---------------------------------------------------------------------------


def test_stage_target_direction_labels_land_on_receiver(tmp_path):
    """label_source='target': provider labels land on every receiver point (D-01)."""
    provider, _ = _clustered(PROVIDER_PER_CLUSTER, seed=1, labelled=True)
    receiver, rcv_cluster = _clustered(RECEIVER_PER_CLUSTER, seed=2, labelled=False)
    estep = _estep(receiver["pos"], provider["pos"])
    assert estep.pmat.shape == (27, 42)

    stage = LabelTransferStage(_stage_cfg(tmp_path, "target"))
    result = stage.run(
        {0: provider}, {0: receiver}, dict(LT_PARAMS),
        align_result=_align_result({0: receiver}, {0: estep}),
    )
    out = result.transferred_labels
    assert sorted(out) == [0]
    assert out[0].shape == (27,)
    assert _values(out[0]) <= set(PROVIDER_CLASSES)
    assert _cluster_accuracy(out[0], rcv_cluster) >= 0.95


def test_stage_target_direction_multi_frame_keys(tmp_path):
    """Per-frame posteriors keyed by the receiver (== target) keys; per-frame counts honoured."""
    keys = (3, 4, 5)
    per_cluster = {3: 9, 4: 8, 5: 10}  # 27, 24, 30 receiver points
    provider, receiver, esteps, clusters = {}, {}, {}, {}
    for i, k in enumerate(keys):
        provider[k], _ = _clustered(PROVIDER_PER_CLUSTER, seed=10 + i, labelled=True)
        receiver[k], clusters[k] = _clustered(per_cluster[k], seed=20 + i, labelled=False)
        esteps[k] = _estep(receiver[k]["pos"], provider[k]["pos"])

    stage = LabelTransferStage(_stage_cfg(tmp_path, "target"))
    result = stage.run(provider, receiver, dict(LT_PARAMS), align_result=_align_result(receiver, esteps))
    out = result.transferred_labels
    assert sorted(out) == list(keys)
    for k in keys:
        assert out[k].shape == (3 * per_cluster[k],), k
        assert _values(out[k]) <= set(PROVIDER_CLASSES), k
        assert _cluster_accuracy(out[k], clusters[k]) >= 0.95, k


def test_stage_default_direction_unchanged(tmp_path):
    """label_source='source' (default): transposed posterior, aligned source -> target."""
    aligned_src, _ = _clustered(RECEIVER_PER_CLUSTER, seed=3, labelled=True)  # 27 labelled
    target, tgt_cluster = _clustered(PROVIDER_PER_CLUSTER, seed=4, labelled=False)  # 42
    estep = _estep(aligned_src["pos"], target["pos"])
    assert estep.pmat.shape == (27, 42)

    stage = LabelTransferStage(_stage_cfg(tmp_path, "source"))
    result = stage.run(
        {0: aligned_src}, {0: target}, dict(LT_PARAMS),
        align_result=_align_result({0: aligned_src}, {0: estep}),
    )
    out = result.transferred_labels[0]
    assert out.shape == (42,)
    assert _values(out) <= set(PROVIDER_CLASSES)
    assert _cluster_accuracy(out, tgt_cluster) >= 0.95


def test_stage_target_direction_zero_mass_row_falls_back_to_nearest(tmp_path):
    """WR-02: an isolated zero-mass receiver row gets the nearest provider label and a flag.

    One underflowing point (far from every provider point) must not abort the
    whole trajectory; the other rows keep the posterior-weighted vote.
    """
    provider, _ = _clustered(PROVIDER_PER_CLUSTER, seed=5, labelled=True)
    receiver, rcv_cluster = _clustered(RECEIVER_PER_CLUSTER, seed=6, labelled=False)
    pos = receiver["pos"].clone()
    pos[0] = torch.tensor([1e4, 1e4, 1e4])
    receiver = zRegPointCloud(pos=pos)
    estep = _estep(pos, provider["pos"], sigma2=1.0)
    row_sums = estep.pmat.sum(dim=1)
    assert row_sums[0].item() == 0.0
    assert bool((row_sums[1:] > 0).all())

    stage = LabelTransferStage(_stage_cfg(tmp_path, "target"))
    result = stage.run(
        {7: provider}, {7: receiver}, dict(LT_PARAMS),
        align_result=_align_result({7: receiver}, {7: estep}),
    )
    out = result.transferred_labels[7]
    assert out.shape == (pos.shape[0],)
    nearest = torch.cdist(pos[:1], provider["pos"]).argmin().item()
    assert out[0].item() == provider["label"][nearest].item()
    assert _values(out) <= set(PROVIDER_CLASSES)
    assert _cluster_accuracy(out[1:], rcv_cluster[1:]) >= 0.95
    assert len(result.flags) == 1
    assert "frame 7" in result.flags[0] and "1 of" in result.flags[0]


def test_stage_target_direction_all_zero_mass_frame_raises(tmp_path):
    """A frame in which every receiver row has zero mass still fails loudly naming the frame (D-05)."""
    provider, _ = _clustered(PROVIDER_PER_CLUSTER, seed=5, labelled=True)
    receiver, _ = _clustered(RECEIVER_PER_CLUSTER, seed=6, labelled=False)
    pos = receiver["pos"].clone() + 1e4
    receiver = zRegPointCloud(pos=pos)
    estep = _estep(pos, provider["pos"], sigma2=1.0)
    assert bool((estep.pmat.sum(dim=1) == 0).all())

    stage = LabelTransferStage(_stage_cfg(tmp_path, "target"))
    with pytest.raises(ValueError, match=r"posterior.*every receiver point.*frame 7"):
        stage.run(
            {7: provider}, {7: receiver}, dict(LT_PARAMS),
            align_result=_align_result({7: receiver}, {7: estep}),
        )


# ---------------------------------------------------------------------------
# Runner level (real AlignmentStage + LabelTransferStage, EvaluationRunner._run_single)
# ---------------------------------------------------------------------------


def _trajectories(n_frames, seed):
    """Receiver-like (unlabelled, 27 pts) source and provider-like (labelled, 42 pts) target."""
    source, target = {}, {}
    for k in range(n_frames):
        source[k], _ = _clustered(RECEIVER_PER_CLUSTER, seed=seed + 2 * k, labelled=False)
        target[k], _ = _clustered(PROVIDER_PER_CLUSTER, seed=seed + 2 * k + 1, labelled=True)
    return source, target


def _assert_target_direction_result(result):
    align = result["align"]
    assert align.estep_results, "CPD posterior must be captured"
    assert result["label_receiver"] is align.aligned_cloud
    transferred = result["label"].transferred_labels
    assert transferred
    for k, labels in transferred.items():
        assert labels.shape[0] == align.aligned_cloud[k]["pos"].shape[0], k
        assert _values(labels) <= set(PROVIDER_CLASSES), k
    assert any(f.startswith("f1 unavailable") for f in result["sanity_flags"])


def test_cpd_weighted_real_yaml_has_required_params():
    """cpd_weighted/real.yaml carries every LabelTransferStage.REQUIRED_PARAMS key (cycle-3 HIGH)."""
    cfg = EvalConfig.from_yaml(REAL_YAML)
    missing = [p for p in LabelTransferStage.REQUIRED_PARAMS if p not in cfg.default_params]
    assert missing == []


def test_run_single_cpd_weighted_real_yaml_nonrigid(tmp_path):
    """cpd_weighted/real.yaml settings verbatim through _run_single with real non-rigid CPD."""
    cfg = EvalConfig.from_yaml(REAL_YAML).model_copy(
        update={"output_dir": str(tmp_path), "save_plots": False}
    )
    assert cfg.label_source == "target"
    assert cfg.label_transfer_method == "cpd_weighted"
    assert cfg.pipeline_mode == "paired"
    assert cfg.run_alignment is True
    assert cfg.default_params["cpd_penalty"] == "nonrigid"

    params = dict(cfg.default_params)  # verbatim: no keys added or overridden
    source, target = _trajectories(n_frames=8, seed=100)
    runner = EvaluationRunner(cfg, params)
    runner.factory = DataFactory(cfg)

    t0 = time.perf_counter()
    result = runner._run_single(source, target, params)
    elapsed = time.perf_counter() - t0
    print(f"real.yaml non-rigid _run_single runtime: {elapsed:.2f}s")

    _assert_target_direction_result(result)


def test_run_single_cpd_weighted_target_direction_end_to_end(tmp_path):
    """Rigid CPD alignment, provider (target) labels transferred onto the aligned source."""
    cfg = EvalConfig(
        data_path=str(tmp_path / "kobitski.tracklets"),
        target_data_path=str(tmp_path / "shah.csv"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        label_source="target",
        run_alignment=True,
        alignment_method="cpd",
        run_label_transfer=True,
        label_transfer_method="cpd_weighted",
        save_plots=False,
    )
    params = {**RIGID_ALIGN_PARAMS, **{k: v for k, v in LT_PARAMS.items() if k != "method"}}
    source, target = _trajectories(n_frames=3, seed=200)
    runner = EvaluationRunner(cfg, params)
    runner.factory = DataFactory(cfg)

    result = runner._run_single(source, target, params)
    _assert_target_direction_result(result)
