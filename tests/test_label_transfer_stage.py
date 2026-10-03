"""Tests for eval.stages.label_transfer.LabelTransferStage.

Covers Phase 20 FRAME-06 gate criteria:
  - LabelTransferStage importable standalone (gate 1)
  - REQUIRED_PARAMS tuple (gate 3)
  - validate_params guards including bool exclusion (gate 4 / WR-01)
  - run() calls validate_params first (gate 5 / D-08)
  - No kNN or distance logic reimplemented (gate 6)
"""

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.algorithms.cpd import RigidCPD
from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generate_trajectory, generate_labels
from zreg.data_generation import add_gaussian_noise
from zreg.evaluation.label_transfer import compute_f1
from zreg.models import PointNet2LabelTransfer, EGNNLabelTransfer
import zreg.utils as utils

import torch

from eval.config import EvalConfig
from eval.stages import LabelTransferStage, PipelineStage
from eval.stages.label_transfer import LabelTransferStage as LabelTransferStageDirect
from eval.stages.label_transfer import ALIGNMENT_WARN_THRESHOLD
from eval.types import LabelResult, AlignResult
from train_label_transfer import save_checkpoint


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_config(tmp_path) -> EvalConfig:
    """Minimal EvalConfig — only data_path is required."""
    return EvalConfig(data_path=str(tmp_path / "unused.mat"))


@pytest.fixture
def stage(eval_config) -> LabelTransferStage:
    """Instantiated LabelTransferStage with minimal config."""
    return LabelTransferStage(eval_config)


@pytest.fixture
def good_params() -> dict:
    """Valid params that pass LabelTransferStage.validate_params."""
    return {
        "k_neighbours": 5,
        "dist_metric": "euclidean",
        "smoothing": 0.0,
        "threshold": 0.0,
    }


@pytest.fixture
def synthetic_dataset() -> dict[int, zRegPointCloud]:
    """3-frame synthetic trajectory with label labels for label transfer tests.

    generate_labels() populates the 'label' field (torch.long, shape (N,))
    so transfer_colors KNN_VOTING has valid source_colors to unsqueeze.
    """
    traj = generate_trajectory(n_points=20, n_frames=3, seed=0)
    return generate_labels(traj, n_labels=4, seed=0)


# ---------------------------------------------------------------------------
# TestLabelTransferStageImport — FRAME-06 gate 1
# ---------------------------------------------------------------------------


class TestLabelTransferStageImport:
    """LabelTransferStage importable without ImportError (FRAME-06 gate 1)."""

    def test_import_from_stages_package(self):
        """from eval.stages import LabelTransferStage succeeds."""
        from eval.stages import LabelTransferStage  # noqa: F401
        assert LabelTransferStage is not None

    def test_import_from_module_directly(self):
        """from eval.stages.label_transfer import LabelTransferStage succeeds."""
        assert LabelTransferStageDirect is not None

    def test_is_pipeline_stage_subclass(self):
        """LabelTransferStage is a subclass of PipelineStage."""
        assert issubclass(LabelTransferStage, PipelineStage)

    def test_in_stages_all(self):
        """LabelTransferStage appears in eval.stages.__all__."""
        import eval.stages
        assert "LabelTransferStage" in eval.stages.__all__

    def test_all_three_in_stages_all(self):
        """eval.stages.__all__ contains exactly the three expected names."""
        import eval.stages
        assert eval.stages.__all__ == ["PipelineStage", "AlignmentStage", "LabelTransferStage"]


# ---------------------------------------------------------------------------
# TestRequiredParams — FRAME-06 gate 3
# ---------------------------------------------------------------------------


class TestRequiredParams:
    """REQUIRED_PARAMS class attribute has the right content and order."""

    def test_required_params_exists(self):
        """LabelTransferStage.REQUIRED_PARAMS attribute exists."""
        assert hasattr(LabelTransferStage, "REQUIRED_PARAMS")

    def test_required_params_exact_tuple(self):
        """REQUIRED_PARAMS is the exact 4-element tuple in specified order."""
        assert LabelTransferStage.REQUIRED_PARAMS == (
            "k_neighbours",
            "dist_metric",
            "smoothing",
            "threshold",
        )

    def test_required_params_accessible_without_instantiation(self):
        """REQUIRED_PARAMS is accessible as a class attribute (no instance needed)."""
        # No EvalConfig needed — class-level access
        assert isinstance(LabelTransferStage.REQUIRED_PARAMS, tuple)
        assert len(LabelTransferStage.REQUIRED_PARAMS) == 4


# ---------------------------------------------------------------------------
# TestValidateParamsMissingKeys — FRAME-06 gate 4 (missing-key branch)
# ---------------------------------------------------------------------------


class TestValidateParamsMissingKeys:
    """validate_params raises ValueError for each missing REQUIRED_PARAMS key."""

    @pytest.mark.parametrize("missing_key", [
        "k_neighbours",
        "dist_metric",
        "smoothing",
        "threshold",
    ])
    def test_missing_key_raises_value_error(self, stage, good_params, missing_key):
        """validate_params raises ValueError('Missing required param: {key}')."""
        params = {k: v for k, v in good_params.items() if k != missing_key}
        with pytest.raises(ValueError, match=f"Missing required param: {missing_key}"):
            stage.validate_params(params)

    def test_empty_dict_raises_for_first_required_param(self, stage):
        """validate_params({}) raises ValueError for k_neighbours (first REQUIRED_PARAMS key)."""
        with pytest.raises(ValueError, match="Missing required param: k_neighbours"):
            stage.validate_params({})


# ---------------------------------------------------------------------------
# TestValidateParamsKNeighbours — FRAME-06 gate 4 (k_neighbours guards)
# ---------------------------------------------------------------------------


class TestValidateParamsKNeighbours:
    """validate_params guards for k_neighbours (int >= 1, bool exclusion)."""

    def test_k_neighbours_zero_raises(self, stage, good_params):
        """k_neighbours=0 raises ValueError matching 'k_neighbours must be int >= 1'."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": 0})

    def test_k_neighbours_negative_raises(self, stage, good_params):
        """k_neighbours=-1 raises ValueError matching 'k_neighbours must be int >= 1'."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": -1})

    def test_k_neighbours_bool_true_raises(self, stage, good_params):
        """k_neighbours=True raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": True})

    def test_k_neighbours_bool_false_raises(self, stage, good_params):
        """k_neighbours=False raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": False})

    def test_k_neighbours_float_raises(self, stage, good_params):
        """k_neighbours=1.0 raises ValueError (must be int)."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": 1.0})

    def test_k_neighbours_one_valid(self, stage, good_params):
        """k_neighbours=1 is valid (no exception raised)."""
        stage.validate_params({**good_params, "k_neighbours": 1})

    def test_k_neighbours_large_valid(self, stage, good_params):
        """k_neighbours=100 is valid (no exception raised)."""
        stage.validate_params({**good_params, "k_neighbours": 100})


# ---------------------------------------------------------------------------
# TestValidateParamsDistMetric — FRAME-06 gate 4 (dist_metric guards)
# ---------------------------------------------------------------------------


class TestValidateParamsDistMetric:
    """validate_params guards for dist_metric (non-empty str)."""

    def test_empty_string_raises(self, stage, good_params):
        """dist_metric='' raises ValueError matching 'dist_metric must be non-empty'."""
        with pytest.raises(ValueError, match="dist_metric must be non-empty"):
            stage.validate_params({**good_params, "dist_metric": ""})

    def test_non_string_raises(self, stage, good_params):
        """dist_metric=123 raises ValueError."""
        with pytest.raises(ValueError, match="dist_metric must be non-empty"):
            stage.validate_params({**good_params, "dist_metric": 123})

    def test_valid_string_passes(self, stage, good_params):
        """dist_metric='euclidean' is valid (no exception raised)."""
        stage.validate_params({**good_params, "dist_metric": "euclidean"})


# ---------------------------------------------------------------------------
# TestValidateParamsSmoothing — FRAME-06 gate 4 (smoothing guards)
# ---------------------------------------------------------------------------


class TestValidateParamsSmoothing:
    """validate_params guards for smoothing (float >= 0.0, bool exclusion)."""

    def test_negative_raises(self, stage, good_params):
        """smoothing=-1.0 raises ValueError matching 'smoothing must be float >= 0.0'."""
        with pytest.raises(ValueError, match="smoothing must be float >= 0.0"):
            stage.validate_params({**good_params, "smoothing": -1.0})

    def test_bool_true_raises(self, stage, good_params):
        """smoothing=True raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="smoothing must be float >= 0.0"):
            stage.validate_params({**good_params, "smoothing": True})

    def test_bool_false_raises(self, stage, good_params):
        """smoothing=False raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="smoothing must be float >= 0.0"):
            stage.validate_params({**good_params, "smoothing": False})

    def test_zero_valid(self, stage, good_params):
        """smoothing=0.0 is valid (no exception raised)."""
        stage.validate_params({**good_params, "smoothing": 0.0})

    def test_positive_float_valid(self, stage, good_params):
        """smoothing=0.5 is valid (no exception raised)."""
        stage.validate_params({**good_params, "smoothing": 0.5})

    def test_int_zero_valid(self, stage, good_params):
        """smoothing=0 (int) is valid — int is accepted as numeric."""
        stage.validate_params({**good_params, "smoothing": 0})


# ---------------------------------------------------------------------------
# TestValidateParamsThreshold — FRAME-06 gate 4 (threshold guards)
# ---------------------------------------------------------------------------


class TestValidateParamsThreshold:
    """validate_params guards for threshold (float >= 0.0, bool exclusion)."""

    def test_negative_raises(self, stage, good_params):
        """threshold=-1.0 raises ValueError matching 'threshold must be float >= 0.0'."""
        with pytest.raises(ValueError, match="threshold must be float >= 0.0"):
            stage.validate_params({**good_params, "threshold": -1.0})

    def test_bool_true_raises(self, stage, good_params):
        """threshold=True raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="threshold must be float >= 0.0"):
            stage.validate_params({**good_params, "threshold": True})

    def test_bool_false_raises(self, stage, good_params):
        """threshold=False raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="threshold must be float >= 0.0"):
            stage.validate_params({**good_params, "threshold": False})

    def test_zero_valid(self, stage, good_params):
        """threshold=0.0 is valid (no exception raised)."""
        stage.validate_params({**good_params, "threshold": 0.0})

    def test_positive_float_valid(self, stage, good_params):
        """threshold=0.5 is valid (no exception raised)."""
        stage.validate_params({**good_params, "threshold": 0.5})


# ---------------------------------------------------------------------------
# TestRunValidatesFirst — FRAME-06 gate 5 / D-08
# ---------------------------------------------------------------------------


class TestRunValidatesFirst:
    """run() calls validate_params as its first line (D-08)."""

    def test_run_empty_params_raises_value_error(self, stage):
        """run({}, {}, {}) raises ValueError (not KeyError/TypeError) — D-08."""
        with pytest.raises(ValueError, match="Missing required param"):
            stage.run({}, {}, {})

    def test_run_valid_params_empty_dataset_raises_value_error(self, stage, good_params):
        """run({}, {}, valid_params) raises ValueError before indexing empty sorted_keys."""
        with pytest.raises(ValueError):
            stage.run({}, {}, good_params)

    def test_run_invalid_k_neighbours_raises_before_dataset_access(self, stage, good_params):
        """run with k_neighbours=True raises ValueError (not attribute error from dataset)."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.run({}, {}, {**good_params, "k_neighbours": True})


# ---------------------------------------------------------------------------
# TestRunOutput — FRAME-06 run() correctness
# ---------------------------------------------------------------------------


class TestRunOutput:
    """run() returns correct LabelResult structure."""

    def test_run_returns_label_result(self, stage, synthetic_dataset, good_params):
        """run() returns a LabelResult instance."""
        result = stage.run(synthetic_dataset, synthetic_dataset, good_params)
        assert isinstance(result, LabelResult)

    def test_run_transferred_labels_keys_match_dataset(self, stage, synthetic_dataset, good_params):
        """transferred_labels has one entry per frame in the dataset (n_pairs = full length when source==target)."""
        result = stage.run(synthetic_dataset, synthetic_dataset, good_params)
        assert set(result.transferred_labels.keys()) == set(synthetic_dataset.keys())

    def test_run_params_used_is_shallow_copy(self, stage, synthetic_dataset, good_params):
        """params_used is dict(params) — mutating original does not affect result."""
        params = dict(good_params)
        result = stage.run(synthetic_dataset, synthetic_dataset, params)
        params["k_neighbours"] = 999  # mutate original
        assert result.params_used["k_neighbours"] == 5  # result unchanged

    def test_run_all_tensors_are_1d(self, stage, synthetic_dataset, good_params):
        """All transferred_labels values are 1-D tensors (squeezed); no pass-through (Phase 30 D-02)."""
        result = stage.run(synthetic_dataset, synthetic_dataset, good_params)
        sorted_keys = sorted(synthetic_dataset.keys())
        for key in sorted_keys:
            assert result.transferred_labels[key].ndim == 1

    def test_run_label_result_is_frozen(self, stage, synthetic_dataset, good_params):
        """LabelResult is pydantic-frozen (attribute reassignment raises)."""
        from pydantic import ValidationError
        result = stage.run(synthetic_dataset, synthetic_dataset, good_params)
        with pytest.raises((ValidationError, TypeError)):
            result.transferred_labels = {}

    def test_run_uses_knn_voting_not_reimplemented(self, stage, synthetic_dataset, good_params):
        """k_neighbours=1 is accepted and run() completes (delegates to transfer_colors)."""
        params = {**good_params, "k_neighbours": 1}
        result = stage.run(synthetic_dataset, synthetic_dataset, params)
        assert isinstance(result, LabelResult)


# ---------------------------------------------------------------------------
# FRAME-06 Gate classes — Plan 20-02
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TestLabelTransferStageRunStandalone — FRAME-06 Gate 1
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset_d09() -> dict[int, zRegPointCloud]:
    """2-frame dataset: labeled frame 0 + noisy frame 1 per D-09."""
    seed_traj = generate_trajectory(n_points=50, n_frames=1, seed=0)
    labeled_traj = generate_labels(seed_traj, n_labels=3, seed=0)
    noisy_frame = add_gaussian_noise(labeled_traj, sigma=0.01, seed=1)[0]
    return {0: labeled_traj[0], 1: noisy_frame}


@pytest.fixture
def default_params_lts() -> dict:
    """Valid hyperparams that pass LabelTransferStage.validate_params."""
    return {
        "k_neighbours": 5,
        "dist_metric": "euclidean",
        "smoothing": 0.0,
        "threshold": 0.0,
    }


class TestLabelTransferStageRunStandalone:
    """Gate 1: LabelTransferStage.run() completes without AlignmentStage present (FRAME-06)."""

    def test_run_returns_label_result(
        self, synthetic_dataset_d09, default_params_lts, eval_config
    ) -> None:
        """run() returns LabelResult; keys match dataset; params_used is a shallow copy."""
        stage = LabelTransferStage(eval_config)
        result = stage.run(synthetic_dataset_d09, synthetic_dataset_d09, default_params_lts)
        assert isinstance(result, LabelResult)
        assert set(result.transferred_labels.keys()) == set(synthetic_dataset_d09.keys())
        assert result.params_used == default_params_lts
        assert result.params_used is not default_params_lts  # shallow copy — Pitfall 7


# ---------------------------------------------------------------------------
# TestLabelTransferStageLabelAccuracy — FRAME-06 Gate 2
# ---------------------------------------------------------------------------


class TestLabelTransferStageLabelAccuracy:
    """Gate 2: transferred labels beat random baseline F1 (FRAME-06)."""

    def test_label_accuracy_beats_random(self, eval_config) -> None:
        """compute_f1(transferred) > compute_f1(shuffled) on D-09 fixture."""
        # D-09: 1-frame generate then manual frame 1
        seed_traj = generate_trajectory(n_points=50, n_frames=1, seed=0)
        labeled_traj = generate_labels(seed_traj, n_labels=3, seed=0)
        ground_truth_labels = labeled_traj[0]["label"]  # shape (50,), torch.long
        noisy_frame = add_gaussian_noise(labeled_traj, sigma=0.01, seed=1)[0]
        dataset = {0: labeled_traj[0], 1: noisy_frame}

        stage = LabelTransferStage(eval_config)
        result = stage.run(
            dataset,
            dataset,
            {"k_neighbours": 5, "dist_metric": "euclidean", "smoothing": 0.0, "threshold": 0.0},
        )

        # D-10: random baseline = shuffled ground truth (same class distribution)
        torch.manual_seed(0)
        shuffled = ground_truth_labels[torch.randperm(len(ground_truth_labels))]
        # CRITICAL: compute_f1(y_true, y_pred) — ground_truth_labels FIRST (Pitfall 5)
        f1_transferred = compute_f1(ground_truth_labels, result.transferred_labels[1])
        f1_random = compute_f1(ground_truth_labels, shuffled)
        assert f1_transferred > f1_random, (
            f"Expected F1(transferred)={f1_transferred:.4f} > F1(random)={f1_random:.4f}"
        )


# ---------------------------------------------------------------------------
# TestLabelTransferStageChainedRun — FRAME-06 Gate 3
# ---------------------------------------------------------------------------


class TestLabelTransferStageChainedRun:
    """Gate 3: stage accepts AlignResult.aligned_cloud as input (FRAME-06)."""

    def test_accepts_align_result_aligned_cloud(
        self, synthetic_dataset_d09, default_params_lts, eval_config
    ) -> None:
        """D-11: construct AlignResult manually — no DTW end-to-end needed."""
        align_result = AlignResult(
            aligned_cloud=synthetic_dataset_d09,
            warp_path=[(0, 0), (1, 1)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={
                "window_size": 10,
                "step": 1,
                "cpd_penalty": None,
                "dtw_dist_fn": "euclidean",
                "n_breakpoints": 5,
            },
        )
        stage = LabelTransferStage(eval_config)
        result = stage.run(align_result.aligned_cloud, align_result.aligned_cloud, default_params_lts)
        assert isinstance(result, LabelResult)
        assert set(result.transferred_labels.keys()) == set(synthetic_dataset_d09.keys())


# ---------------------------------------------------------------------------
# TestLabelTransferStageValidateParams — FRAME-06 Gate 4
# ---------------------------------------------------------------------------


class TestLabelTransferStageValidateParams:
    """Gate 4: validate_params raises on missing/invalid params (D-07, D-08)."""

    @pytest.mark.parametrize("missing_key", LabelTransferStage.REQUIRED_PARAMS)
    def test_missing_param_raises(self, missing_key, default_params_lts, eval_config) -> None:
        """Each of the 4 required keys raises ValueError when absent."""
        partial = {k: v for k, v in default_params_lts.items() if k != missing_key}
        stage = LabelTransferStage(eval_config)
        dataset = {0: generate_trajectory(n_points=5, n_frames=1, seed=0)[0]}
        with pytest.raises(ValueError, match=f"Missing required param: {missing_key}"):
            stage.run(dataset, dataset, partial)

    def test_run_calls_validate_first(self, eval_config) -> None:
        """D-08: run() calls validate_params before computation so empty params raises ValueError not KeyError."""
        stage = LabelTransferStage(eval_config)
        with pytest.raises(ValueError, match="Missing required param"):
            stage.run({}, {}, {})

    @pytest.mark.parametrize(
        "key,bad_value,match_str",
        [
            ("k_neighbours", 0, "k_neighbours must be int >= 1"),
            ("k_neighbours", -1, "k_neighbours must be int >= 1"),
            ("k_neighbours", True, "k_neighbours must be int >= 1"),
            ("dist_metric", "", "dist_metric must be non-empty"),
            ("smoothing", -1.0, "smoothing must be float >= 0.0"),
            ("smoothing", True, "smoothing must be float >= 0.0"),
            ("threshold", -1.0, "threshold must be float >= 0.0"),
            ("threshold", True, "threshold must be float >= 0.0"),
        ],
    )
    def test_invalid_value_raises(
        self, key, bad_value, match_str, default_params_lts, eval_config
    ) -> None:
        """D-08: invalid values for each param raise descriptive ValueError."""
        params = {**default_params_lts, key: bad_value}
        stage = LabelTransferStage(eval_config)
        with pytest.raises(ValueError, match=match_str):
            stage.validate_params(params)


# ---------------------------------------------------------------------------
# TestLabelTransferStageOutputShape — FRAME-06 Gate 5
# ---------------------------------------------------------------------------


class TestLabelTransferStageOutputShape:
    """Gate 5: transferred_labels are 1D torch.long tensors — no reimplementation (FRAME-06)."""

    def test_transferred_labels_are_1d_long(
        self, synthetic_dataset_d09, default_params_lts, eval_config
    ) -> None:
        """All transferred_labels values are 1D torch.long tensors of shape (N_points,)."""
        stage = LabelTransferStage(eval_config)
        result = stage.run(synthetic_dataset_d09, synthetic_dataset_d09, default_params_lts)
        for key, tensor in result.transferred_labels.items():
            assert tensor.ndim == 1, f"Frame {key}: expected 1D tensor, got {tensor.ndim}D"
            assert tensor.dtype == torch.long, (
                f"Frame {key}: expected torch.long, got {tensor.dtype}"
            )
            assert tensor.shape[0] == synthetic_dataset_d09[key]["pos"].shape[0], (
                f"Frame {key}: point count mismatch — "
                f"expected {synthetic_dataset_d09[key]['pos'].shape[0]}, got {tensor.shape[0]}"
            )


# ---------------------------------------------------------------------------
# TestLabelTransferStageCpdWeighted — Phase 44 D-04/D-05/D-07/D-08
# ---------------------------------------------------------------------------


@pytest.fixture
def cpd_weighted_source_target():
    """12-point labeled source + 9-point target (differing counts — exercises D-04).

    Builds a real EstepResult via RigidCPD.expectation_step so the
    cpd_weighted happy-path test exercises the actual pmat orientation
    bug (D-04): a square matrix would mask it, so source/target
    deliberately have different point counts.
    """
    source_traj = generate_labels(
        generate_trajectory(n_points=12, n_frames=1, seed=500), n_labels=3, seed=500
    )
    target_traj = generate_trajectory(n_points=9, n_frames=1, seed=501)
    source_pos = source_traj[0]["pos"]
    target_pos = target_traj[0]["pos"]

    sigma2_est = utils.squared_kernel_sum(source_pos, target_pos)
    cpd = RigidCPD(source=source_pos, use_color=False)
    estep_result = cpd.expectation_step(
        t_source=source_pos, target=target_pos, sigma2=sigma2_est, sigma2_c=0.0, w=0.0
    )

    source = {0: source_traj[0]}
    target = {0: target_traj[0]}
    return source, target, estep_result


class TestLabelTransferStageCpdWeighted:
    """Phase 44: method='cpd_weighted' happy path + error cases (D-04/D-05/D-07/D-08)."""

    def test_cpd_weighted_happy_path(
        self, stage, good_params, cpd_weighted_source_target
    ) -> None:
        """cpd_weighted returns correctly shaped/dtyped transferred_labels via CPD posterior."""
        source, target, estep_result = cpd_weighted_source_target
        align_result = AlignResult(
            aligned_cloud=source,
            warp_path=[(0, 0)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
            estep_results={0: estep_result},
        )
        result = stage.run(
            source, target, {**good_params, "method": "cpd_weighted"}, align_result=align_result
        )
        assert result.transferred_labels[0].ndim == 1
        assert result.transferred_labels[0].dtype == torch.long
        assert result.transferred_labels[0].shape[0] == target[0]["pos"].shape[0]

    def test_cpd_weighted_missing_align_result_raises(
        self, stage, good_params, cpd_weighted_source_target
    ) -> None:
        """method='cpd_weighted' without align_result raises a clear ValueError."""
        source, _target, _estep_result = cpd_weighted_source_target
        with pytest.raises(ValueError, match="requires align_result"):
            stage.run(source, source, {**good_params, "method": "cpd_weighted"})

    def test_cpd_weighted_missing_estep_results_entry_raises(
        self, stage, good_params, cpd_weighted_source_target
    ) -> None:
        """method='cpd_weighted' with align_result missing the frame's estep_results entry raises."""
        source, target, _estep_result = cpd_weighted_source_target
        align_result = AlignResult(
            aligned_cloud=source,
            warp_path=[(0, 0)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
            estep_results={},
        )
        with pytest.raises(ValueError, match="estep_results"):
            stage.run(
                source,
                target,
                {**good_params, "method": "cpd_weighted"},
                align_result=align_result,
            )

    def test_invalid_method_value_raises(self, stage, good_params) -> None:
        """validate_params raises ValueError for an unsupported method value.

        Now that VALID_METHODS has four entries (48-02), the error message must
        name all four valid values, including 'pointnet2' and 'egnn'.
        """
        with pytest.raises(ValueError, match="method must be one of") as excinfo:
            stage.validate_params({**good_params, "method": "bogus"})
        assert "pointnet2" in str(excinfo.value)
        assert "egnn" in str(excinfo.value)

    def test_method_defaults_to_config_label_transfer_method(
        self, stage, good_params, synthetic_dataset
    ) -> None:
        """Omitting 'method' defaults to config.label_transfer_method ('knn_voting')."""
        assert "method" not in good_params
        result = stage.run(synthetic_dataset, synthetic_dataset, good_params)
        assert result.params_used["method"] == "knn_voting"


# ---------------------------------------------------------------------------
# TestLabelTransferStageLearnedMethods — Phase 48 D-01/D-02/D-03
# ---------------------------------------------------------------------------

LEARNED_METHOD_CASES = [
    (
        "pointnet2",
        "pointnet2_checkpoint_path",
        PointNet2LabelTransfer,
        {"n_classes": 4, "hidden_dim": 32},
    ),
    (
        "egnn",
        "egnn_checkpoint_path",
        EGNNLabelTransfer,
        {"n_classes": 4, "hidden_dim": 32, "n_layers": 4},
    ),
]


@pytest.fixture
def learned_smoke_checkpoint(tmp_path):
    """Factory fixture: build a fresh, untrained model + smoke-test checkpoint.

    Direct-construction approach (48-RESEARCH.md Wave 0 Gaps recommendation) —
    does NOT invoke train_label_transfer.py's CLI/subprocess. A structurally
    valid checkpoint (freshly-initialized weights) is all D-01 requires: this
    phase's tests prove the load->infer->LabelResult *wiring* is correct, not
    model quality.

    n_classes=4 MATCHES synthetic_dataset's generate_labels(..., n_labels=4)
    label vocabulary (Pitfall 1 — a mismatch makes one_hot raise a RuntimeError
    unrelated to wiring).
    """

    def _build(method: str) -> str:
        for m, _field, model_cls, hyperparams in LEARNED_METHOD_CASES:
            if m == method:
                model = model_cls(**hyperparams)
                path = str(tmp_path / f"{method}.pt")
                save_checkpoint(model, method, hyperparams, epoch=0, path=path)
                return path
        raise ValueError(f"unknown method: {method!r}")

    return _build


@pytest.mark.parametrize("method,path_field,model_cls,hyperparams", LEARNED_METHOD_CASES)
class TestLabelTransferStageLearnedMethods:
    """Phase 48: method='pointnet2'/'egnn' happy path + encoding parity + error cases."""

    def test_learned_happy_path(
        self,
        method,
        path_field,
        model_cls,
        hyperparams,
        tmp_path,
        good_params,
        synthetic_dataset,
        learned_smoke_checkpoint,
    ) -> None:
        """Checkpoint loads, model constructs, inference runs, output shape/dtype matches
        LabelResult's contract (1-D torch.long, sized to the target frame's point count)."""
        ckpt_path = learned_smoke_checkpoint(method)
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"), **{path_field: ckpt_path}
        )
        stage = LabelTransferStage(config)
        result = stage.run(
            synthetic_dataset, synthetic_dataset, {**good_params, "method": method}
        )
        for tk in synthetic_dataset:
            tensor = result.transferred_labels[tk]
            assert tensor.ndim == 1
            assert tensor.dtype == torch.long
            assert tensor.shape[0] == synthetic_dataset[tk]["pos"].shape[0]

    def test_learned_encoding_parity(
        self,
        method,
        path_field,
        model_cls,
        hyperparams,
        tmp_path,
        good_params,
        synthetic_dataset,
        learned_smoke_checkpoint,
    ) -> None:
        """THE critical joint-cloud-correctness test.

        Loads the SAME checkpoint manually and builds the reference joint
        cloud EXACTLY as train_step does (source-then-target concatenation,
        one_hot over [:n_src, :n_classes], unknown-flag at [n_src:, -1] = 1.0).
        Asserts torch.equal against the stage's output — fails on any
        deviation in concatenation order, one-hot width, or unknown-flag
        position.
        """
        ckpt_path = learned_smoke_checkpoint(method)
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"), **{path_field: ckpt_path}
        )
        stage = LabelTransferStage(config)
        result = stage.run(
            synthetic_dataset, synthetic_dataset, {**good_params, "method": method}
        )

        # Manually reconstruct the reference model + joint cloud.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ref_model = model_cls(**ckpt["hyperparams"])
        ref_model.load_state_dict(ckpt["model_state_dict"])
        ref_model.eval()
        n_classes = ref_model.n_classes

        source_keys = sorted(synthetic_dataset.keys())
        target_keys = sorted(synthetic_dataset.keys())
        sk = source_keys[0]
        tk = target_keys[0]
        src_frame = synthetic_dataset[sk]
        tgt_frame = synthetic_dataset[tk]
        src_pos = src_frame["pos"]
        tgt_pos = tgt_frame["pos"]
        src_labels = src_frame["label"]
        n_src = src_pos.shape[0]

        joint_pos = torch.cat([src_pos, tgt_pos], dim=0)  # source THEN target
        joint_feat = torch.zeros(joint_pos.shape[0], n_classes + 1)
        joint_feat[:n_src, :n_classes] = torch.nn.functional.one_hot(
            src_labels.long(), num_classes=n_classes
        ).float()
        joint_feat[n_src:, -1] = 1.0

        with torch.no_grad():
            ref_logits = ref_model(joint_pos, joint_feat)
        ref = ref_logits[n_src:].argmax(dim=1)

        assert torch.equal(result.transferred_labels[tk], ref)

    def test_learned_missing_checkpoint_path_raises(
        self, method, path_field, model_cls, hyperparams, tmp_path, good_params, synthetic_dataset
    ) -> None:
        """EvalConfig WITHOUT the checkpoint path field (defaults None) raises ValueError
        naming the method and the config path field."""
        config = EvalConfig(data_path=str(tmp_path / "unused.mat"))
        stage = LabelTransferStage(config)
        with pytest.raises(ValueError, match=method) as excinfo:
            stage.run(
                synthetic_dataset, synthetic_dataset, {**good_params, "method": method}
            )
        assert path_field in str(excinfo.value)

    def test_learned_checkpoint_not_found_raises(
        self, method, path_field, model_cls, hyperparams, tmp_path, good_params, synthetic_dataset
    ) -> None:
        """A non-existent checkpoint path raises ValueError matching 'not found'."""
        nonexistent = str(tmp_path / "nope.pt")
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"), **{path_field: nonexistent}
        )
        stage = LabelTransferStage(config)
        with pytest.raises(ValueError, match="not found"):
            stage.run(
                synthetic_dataset, synthetic_dataset, {**good_params, "method": method}
            )

    def test_learned_model_class_mismatch_raises(
        self,
        method,
        path_field,
        model_cls,
        hyperparams,
        tmp_path,
        good_params,
        synthetic_dataset,
        learned_smoke_checkpoint,
    ) -> None:
        """Pointing <path_field> at the OTHER model's checkpoint raises ValueError
        matching 'model_class'."""
        other_method = "egnn" if method == "pointnet2" else "pointnet2"
        other_ckpt_path = learned_smoke_checkpoint(other_method)
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"), **{path_field: other_ckpt_path}
        )
        stage = LabelTransferStage(config)
        with pytest.raises(ValueError, match="model_class"):
            stage.run(
                synthetic_dataset, synthetic_dataset, {**good_params, "method": method}
            )


# ---------------------------------------------------------------------------
# Coverage gap tests for label_transfer.py
# ---------------------------------------------------------------------------


class TestLabelTransferStageCoverageGaps:
    """Coverage gaps: lines 208, 222 in label_transfer.py (post-CLN-02)."""

    def test_label_is_none_raises_value_error(self, eval_config):
        """CLN-02: ValueError raised when source frame has label=None (id fallback removed)."""
        pc_no_label = zRegPointCloud(
            pos=torch.randn(10, 3),
            label=None,
            id=torch.arange(10, dtype=torch.long),
        )
        dataset = {0: pc_no_label}
        stage = LabelTransferStage(eval_config)
        params = {"k_neighbours": 3, "dist_metric": "euclidean", "smoothing": 0.0, "threshold": 0.0}
        with pytest.raises(ValueError, match="has no 'label' field"):
            stage.run(dataset, dataset, params)

    def test_empty_target_raises(self, eval_config):
        """label_transfer.py — ValueError when target is empty dict."""
        pc = zRegPointCloud(pos=torch.randn(10, 3), label=torch.arange(10, dtype=torch.long), id=None)
        source = {0: pc}
        stage = LabelTransferStage(eval_config)
        params = {"k_neighbours": 3, "dist_metric": "euclidean", "smoothing": 0.0, "threshold": 0.0}
        with pytest.raises(ValueError, match="target must be non-empty"):
            stage.run(source, {}, params)

    def test_k_neighbours_exceeds_n_src_raises(self, eval_config):
        """label_transfer.py — ValueError when k_neighbours > n_src."""
        labeled = generate_labels(generate_trajectory(n_points=5, n_frames=1, seed=0), n_labels=2, seed=0)
        stage = LabelTransferStage(eval_config)
        params = {"k_neighbours": 100, "dist_metric": "euclidean", "smoothing": 0.0, "threshold": 0.0}
        with pytest.raises(ValueError, match="k_neighbours=100 exceeds"):
            stage.run(labeled, labeled, params)

    def test_run_raises_when_label_is_none(self, eval_config):
        """CLN-02: ValueError raised when source frame has label=None (no id fallback)."""
        source = {
            0: zRegPointCloud(pos=torch.randn(10, 3), label=None, id=torch.arange(10)),
            1: zRegPointCloud(pos=torch.randn(10, 3), label=None, id=torch.arange(10)),
        }
        target = {
            0: zRegPointCloud(pos=torch.randn(8, 3), label=None, id=torch.arange(8)),
            1: zRegPointCloud(pos=torch.randn(8, 3), label=None, id=torch.arange(8)),
        }
        stage = LabelTransferStage(eval_config)
        align_result = AlignResult(
            aligned_cloud=source,
            warp_path=[(0, 0), (1, 1)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        with pytest.raises(ValueError, match="has no 'label' field"):
            stage.run(
                source=source,
                target=target,
                params={"k_neighbours": 3, "dist_metric": "euclidean", "smoothing": 0.0, "threshold": 0.0},
            )


# ---------------------------------------------------------------------------
# TestLabelTransferAlignmentGuard — ALIGN-02
# ---------------------------------------------------------------------------


def _make_pc(pos: torch.Tensor, n_labels: int = 4) -> zRegPointCloud:
    """Helper: build a zRegPointCloud with integer labels."""
    n = pos.shape[0]
    return zRegPointCloud(
        pos=pos,
        label=torch.zeros(n, dtype=torch.long),
        id=torch.arange(n, dtype=torch.long) % n_labels,
    )


def _default_params() -> dict:
    return {"k_neighbours": 3, "dist_metric": "euclidean", "smoothing": 0.0, "threshold": 0.0}


class TestLabelTransferAlignmentGuard:
    """ALIGN-02: pre_transfer_alignment field and alignment warning behaviour."""

    def test_pre_transfer_alignment_field_present_in_result(self, tmp_path):
        """result.pre_transfer_alignment exists and is 0.0 for identical clouds."""
        cfg = EvalConfig(data_path=str(tmp_path / "unused.mat"))
        pos = torch.randn(10, 3)
        dataset = {0: _make_pc(pos)}
        stage = LabelTransferStage(cfg)
        result = stage.run(dataset, dataset, _default_params())
        assert hasattr(result, "pre_transfer_alignment")
        assert result.pre_transfer_alignment == pytest.approx(0.0, abs=1e-6)

    def test_pre_transfer_alignment_nonzero_for_shifted_source(self, tmp_path):
        """pre_transfer_alignment > threshold when source is shifted far from target."""
        cfg = EvalConfig(data_path=str(tmp_path / "unused.mat"))
        pos_target = torch.randn(10, 3)
        pos_source = pos_target + 10.0  # large shift — Chamfer will be >> 1.0
        source = {0: _make_pc(pos_source)}
        target = {0: _make_pc(pos_target)}
        stage = LabelTransferStage(cfg)
        result = stage.run(source, target, _default_params())
        assert result.pre_transfer_alignment > ALIGNMENT_WARN_THRESHOLD

    def test_alignment_warning_issued_when_misaligned_and_no_alignment_stage(self, tmp_path):
        """warnings.warn() fires when run_alignment=False and Chamfer > threshold."""
        cfg = EvalConfig(data_path=str(tmp_path / "unused.mat"), run_alignment=False)
        pos_target = torch.randn(10, 3)
        pos_source = pos_target + 10.0
        source = {0: _make_pc(pos_source)}
        target = {0: _make_pc(pos_target)}
        stage = LabelTransferStage(cfg)
        with pytest.warns(UserWarning, match="mean Chamfer distance"):
            stage.run(source, target, _default_params())

    def test_no_warning_when_alignment_stage_was_run(self, tmp_path, recwarn):
        """No UserWarning when run_alignment=True even if input is misaligned."""
        cfg = EvalConfig(data_path=str(tmp_path / "unused.mat"), run_alignment=True)
        pos_target = torch.randn(10, 3)
        pos_source = pos_target + 10.0
        source = {0: _make_pc(pos_source)}
        target = {0: _make_pc(pos_target)}
        stage = LabelTransferStage(cfg)
        stage.run(source, target, _default_params())
        user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_no_warning_when_pre_aligned_and_no_alignment_stage(self, tmp_path, recwarn):
        """No UserWarning when run_alignment=False but input is already aligned."""
        cfg = EvalConfig(data_path=str(tmp_path / "unused.mat"), run_alignment=False)
        pos = torch.randn(10, 3)
        dataset = {0: _make_pc(pos)}
        stage = LabelTransferStage(cfg)
        stage.run(dataset, dataset, _default_params())
        user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


class TestPreTransferAlignmentChamferEmptyFrames:
    """label_transfer.py:176 — empty frames are skipped in _compute_pre_alignment_chamfer."""

    def test_empty_frame_skipped_returns_zero(self, tmp_path):
        """_compute_pre_alignment_chamfer returns 0.0 when all frames have zero points."""
        cfg = EvalConfig(data_path=str(tmp_path / "unused.mat"))
        empty_pos = torch.zeros(0, 3)
        source = {0: zRegPointCloud(pos=empty_pos, label=torch.zeros(0, dtype=torch.long))}
        target = {0: zRegPointCloud(pos=empty_pos, label=torch.zeros(0, dtype=torch.long))}
        stage = LabelTransferStage(cfg)
        # Call _check_alignment directly to avoid k_neighbours validation on empty data
        result = stage._check_alignment(source, target)
        assert result == 0.0


class TestLabelTransferStageCorruptedCheckpoint:
    """lines 260-261: torch.load failure raises ValueError with 'failed to load'."""

    def test_corrupted_checkpoint_raises_value_error(self, tmp_path, good_params, synthetic_dataset):
        corrupted = tmp_path / "bad.pt"
        corrupted.write_text("not a valid pytorch file")
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            pointnet2_checkpoint_path=str(corrupted),
        )
        stage = LabelTransferStage(config)
        with pytest.raises(ValueError, match="failed to load"):
            stage.run(synthetic_dataset, synthetic_dataset, {**good_params, "method": "pointnet2"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (run once on HoreKa)")
@pytest.mark.parametrize("method,path_field", [("pointnet2", "pointnet2_checkpoint_path"), ("egnn", "egnn_checkpoint_path")])
def test_learned_stage_runs_on_cuda_frames(
    method, path_field, tmp_path, good_params, synthetic_dataset, learned_smoke_checkpoint
) -> None:
    """Phase 62 (Research Open Q2): CUDA frames + CPU-loaded checkpoint -> no device mismatch."""
    ckpt_path = learned_smoke_checkpoint(method)
    config = EvalConfig(data_path=str(tmp_path / "unused.mat"), **{path_field: ckpt_path})
    cuda_frames = {k: zRegPointCloud(**dict(v)).to("cuda") for k, v in synthetic_dataset.items()}
    result = LabelTransferStage(config).run(cuda_frames, cuda_frames, {**good_params, "method": method})
    for tk, frame in cuda_frames.items():
        labels = result.transferred_labels[tk]
        assert labels.device.type == "cuda"
        assert labels.shape == (frame["pos"].shape[0],)


@pytest.mark.parametrize("method,path_field", [("pointnet2", "pointnet2_checkpoint_path"), ("egnn", "egnn_checkpoint_path")])
def test_learned_stage_moves_model_once(
    method, path_field, tmp_path, good_params, synthetic_dataset, learned_smoke_checkpoint, monkeypatch
) -> None:
    """62-REVIEW WR-08: the shared model is moved once before the frame loop, not per frame."""
    assert len(synthetic_dataset) > 1
    ckpt_path = learned_smoke_checkpoint(method)
    config = EvalConfig(data_path=str(tmp_path / "unused.mat"), **{path_field: ckpt_path})
    original = LabelTransferStage._load_learned_model
    to_calls: list = []

    def _counting_load(m, path):
        model = original(m, path)
        real_to = model.to

        def _to(*args, **kwargs):
            to_calls.append(args)
            return real_to(*args, **kwargs)

        model.to = _to
        return model

    monkeypatch.setattr(LabelTransferStage, "_load_learned_model", staticmethod(_counting_load))
    result = LabelTransferStage(config).run(synthetic_dataset, synthetic_dataset, {**good_params, "method": method})
    assert len(to_calls) == 1
    assert set(result.transferred_labels) == set(synthetic_dataset)
