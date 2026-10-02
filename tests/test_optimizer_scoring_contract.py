"""Contract tests for HyperparamOptimizer scoring (Phase 59 NUM-02, NUM-03, NUM-04).

These are *contract* tests: they drive the real ``_score_subsample_pair_multiseed``,
the real ``_objective``, the real ``MetricsEngine.compute_stage_metrics`` and the
real stages/``DataFactory``.  They must never patch the units under test (D-06,
review pattern 3): a mocked scorer is exactly how the multi-seed path came to
score raw tensors against the *unaligned* source view (chamfer 0.0 for a 0 deg
and a 90 deg pair alike) without any test noticing.

The only stub allowed here is data loading (``DataFactory.load_target``) in the
paired-mode test, because the paired path would otherwise need real files.

Covers:
  - NUM-02: multi-seed HPO scores the aligned per-frame dict against the target
    dict, so a misaligned pair scores worse, both in the last-seed metrics and
    in the averaged objective the search strategies rank on.
  - NUM-03: a list-valued seed runs the sanity and dev tiers on ``seed[0]``;
    ``_tier_dataset`` never returns ``None`` on the subsample_pair path.
  - NUM-04 (HPO half): ``_objective`` honours ``label_source`` through the
    shared ``resolve_label_transfer_pair`` helper and zero-fills F1 when it is
    unavailable.
"""

import logging
import math

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.core.dataset import zRegPointCloud

import torch

import optuna  # noqa: F401  (import-order contract: torch -> optuna -> eval)

from eval.config import EvalConfig
from eval.runners.optimizer import HyperparamOptimizer

_OPT_LOGGER = "eval.runners.optimizer"


def _cfg(tmp_path, *, rot=0.0, seed=None, tier="full") -> EvalConfig:
    """Synthetic subsample_pair config used by the multi-seed / list-seed tests."""
    if seed is None:
        seed = [1, 2]
    return EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="synthetic",
        run_alignment=True,
        run_label_transfer=True,
        transform_spec={
            "type": "subsample_pair",
            "synthesize": True,
            "seed": seed,
            "n_classes": 3,
            "n_points": 80,
            "source_fraction": 0.8,
            "target_fraction": 0.8,
            "rotation_deg": rot,
            "rotation_axis": [0.0, 0.0, 1.0],
            "scale_factor": 1.0,
        },
        tier=tier,
        n_trials=1,
        search_strategy="grid",
        search_space={"window_size": [3], "k_neighbours": [3]},
    )


def _trial_failed_records(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "Trial failed" in r.getMessage()]


# ---------------------------------------------------------------------------
# NUM-02: multi-seed scoring measures alignment quality
# ---------------------------------------------------------------------------


def test_multiseed_scores_alignment_quality(tmp_path) -> None:
    """A 90 deg misaligned pair scores strictly worse than a 0 deg pair.

    With the default params (``cpd_penalty=None`` => temporal-only, no spatial
    registration) the 90 deg pair stays misaligned in the scored dict.
    """
    metrics = {}
    returned = {}
    for rot in (0.0, 90.0):
        opt = HyperparamOptimizer(_cfg(tmp_path / f"rot{int(rot)}", rot=rot))
        hist: list = []
        returned[rot] = opt._score_subsample_pair_multiseed(
            {}, dict(opt._default_params), "full", [1, 2], hist
        )
        # The Trial records exactly the float the search strategy receives.
        assert len(hist) == 1
        assert hist[-1].score == returned[rot]
        metrics[rot] = hist[-1].metrics

    m0, m90 = metrics[0.0], metrics[90.0]
    returned_0, returned_90 = returned[0.0], returned[90.0]

    # (a) last-seed metrics: aligned dicts were scored, not raw tensors.
    assert math.isfinite(m0.chamfer_distance)
    assert m0.chamfer_distance > 0.0  # 6c1c37f: exactly 0.0; after 59-04 only: inf
    assert m90.chamfer_distance > m0.chamfer_distance
    assert m90.hausdorff_distance > m0.hausdorff_distance
    assert m90.normalized["chamfer"] < m0.normalized["chamfer"] < 1.0
    assert m0.coverage_flags == []

    # (b) the averaged objective the search actually ranks on.
    assert math.isfinite(returned_0)
    assert math.isfinite(returned_90)
    assert returned_90 < returned_0


# ---------------------------------------------------------------------------
# NUM-03: list-valued seeds run every tier
# ---------------------------------------------------------------------------


def test_list_seed_runs_every_tier_on_seed0(tmp_path, caplog) -> None:
    """A list seed records trials in sanity, dev and full with no swallowed failure."""
    opt = HyperparamOptimizer(_cfg(tmp_path, seed=[1, 2, 3], tier="full"))
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        result = opt.run()
    assert {t.tier for t in result.history} == {"sanity", "dev", "full"}
    assert _trial_failed_records(caplog) == []
    assert opt._tier_dataset("dev") is not None
    # The pre-populated dev/full dataset was generated from seed[0].
    reference = HyperparamOptimizer(_cfg(tmp_path / "ref", seed=1, tier="full"))
    reference._factory.generate_subsample_pair(
        None, {**reference.config.transform_spec, "seed": 1}
    )
    dev_view = opt._tier_dataset("dev")
    ref_view = reference._tier_dataset("dev")
    assert sorted(dev_view) == sorted(ref_view)
    for k in dev_view:
        assert torch.equal(dev_view[k]["pos"], ref_view[k]["pos"])


def test_list_seed_sanity_tier_uses_seed0(tmp_path, caplog) -> None:
    """A list seed with tier='sanity' records the sanity trials (no ValueError swallowed)."""
    opt = HyperparamOptimizer(_cfg(tmp_path, seed=[5, 6], tier="sanity"))
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        result = opt.run()
    # grid over a 1x1 search space -> exactly one sanity trial
    assert len(result.history) == 1
    assert {t.tier for t in result.history} == {"sanity"}
    assert _trial_failed_records(caplog) == []


def test_list_seed_empty_list_raises_in_run(tmp_path) -> None:
    """An empty seed list is rejected loudly by run()'s pre-populate step."""
    opt = HyperparamOptimizer(_cfg(tmp_path, seed=[], tier="dev"))
    with pytest.raises(ValueError, match="must be non-empty"):
        opt.run()


@pytest.mark.parametrize("tier", ["dev", "full"])
def test_tier_dataset_raises_when_not_prepopulated(tmp_path, tier) -> None:
    """_tier_dataset never returns None on the subsample_pair path."""
    opt = HyperparamOptimizer(_cfg(tmp_path, seed=7))
    with pytest.raises(RuntimeError, match="pre-populate"):
        opt._tier_dataset(tier)


# ---------------------------------------------------------------------------
# NUM-04 (HPO half): _objective honours label_source
# ---------------------------------------------------------------------------


def _frames(n_points: int, labels: bool, offset: float) -> dict:
    gen = torch.Generator().manual_seed(n_points)
    out = {}
    for f in range(3):
        pos = torch.rand(n_points, 3, generator=gen) + offset
        label = (
            torch.randint(10, 13, (n_points,), generator=gen, dtype=torch.long)
            if labels
            else None
        )
        out[f] = zRegPointCloud(pos=pos, label=label, id=None)
    return out


def test_objective_paired_target_direction_uses_provider_labels(
    tmp_path, monkeypatch, caplog
) -> None:
    """label_source='target': labels flow target -> source, F1 is zero-filled."""
    cfg = EvalConfig(
        data_path=str(tmp_path / "kobitski.mat"),
        target_data_path=str(tmp_path / "shah.csv"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        label_source="target",
        run_alignment=False,
        run_label_transfer=True,
        label_transfer_method="knn_voting",
        search_strategy="grid",
        search_space={"k_neighbours": [3]},
    )
    source = _frames(25, labels=False, offset=0.0)  # unlabeled Kobitski-like
    target = _frames(40, labels=True, offset=0.0)  # labelled Shah-like
    opt = HyperparamOptimizer(cfg)
    monkeypatch.setattr(opt._factory, "load_target", lambda: target)

    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective({"k_neighbours": 3}, source, "dev", hist)
        # second trial: the F1-unavailable reason is logged once per instance
        opt._objective({"k_neighbours": 3}, source, "dev", hist)

    assert _trial_failed_records(caplog) == []
    assert len(hist) == 2
    assert math.isfinite(score)
    assert hist[0].score == score
    assert hist[0].metrics.f1_score == 0.0
    assert hist[0].metrics.normalized["f1"] == 0.0
    f1_logs = [r for r in caplog.records if "f1 unavailable" in r.getMessage()]
    assert len(f1_logs) == 1


def test_objective_paired_target_without_label_transfer_zero_fills_f1(
    tmp_path, monkeypatch, caplog
) -> None:
    """F1 unavailable and label transfer off: no GT read, unequal counts do not raise."""
    cfg = EvalConfig(
        data_path=str(tmp_path / "kobitski.mat"),
        target_data_path=str(tmp_path / "shah.csv"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        label_source="target",
        run_alignment=True,
        run_label_transfer=False,
        search_strategy="grid",
        search_space={"window_size": [3]},
    )
    source = _frames(25, labels=False, offset=0.0)
    target = _frames(40, labels=True, offset=0.0)
    opt = HyperparamOptimizer(cfg)
    monkeypatch.setattr(opt._factory, "load_target", lambda: target)

    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective({"window_size": 3}, source, "dev", hist)

    assert _trial_failed_records(caplog) == []
    assert len(hist) == 1
    assert math.isfinite(score)
    assert hist[0].metrics.f1_score == 0.0
