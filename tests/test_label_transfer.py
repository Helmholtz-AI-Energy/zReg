"""Tests for color transfer functionality."""

import pytest
import torch

from zreg.label_transfer import transfer_labels as transfer_colors, LabelTransferMethod as ColorTransferMethod
from zreg.label_transfer import transfer_labels, LabelTransferMethod
from zreg.core.dataset import zRegPointCloud


class MockEstepResult:
    """Mock EstepResult for testing."""
    def __init__(self, pmat):
        self.pt1 = pmat.sum(dim=1)  # sum over source for each target
        self.p1 = pmat.sum(dim=0)   # sum over target for each source
        self.px = torch.matmul(pmat, torch.zeros(pmat.shape[1], 3))  # dummy, any pmat shape
        self.n_p = self.p1.sum()
        self.pmat = pmat


@pytest.fixture
def sample_point_clouds():
    """Create sample point clouds for testing."""
    # Source point cloud
    source_pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ], dtype=torch.float32)

    source_colors = torch.tensor([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0]   # Yellow
    ], dtype=torch.float32)

    source_pc = zRegPointCloud(pos=source_pos, label=source_colors)

    # Target point cloud (slightly offset)
    target_pos = torch.tensor([
        [0.1, 0.1, 0.0],  # Close to source[0]
        [0.9, 0.1, 0.0],  # Close to source[1]
        [0.1, 0.9, 0.0],  # Close to source[2]
        [0.9, 0.9, 0.0]   # Close to source[3]
    ], dtype=torch.float32)

    target_pc = zRegPointCloud(pos=target_pos)

    return source_pc, target_pc


def test_transfer_colors_nearest_neighbor_zreg(sample_point_clouds):
    """Test nearest neighbor color transfer with zRegPointCloud inputs."""
    source_pc, target_pc = sample_point_clouds

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.NEAREST_NEIGHBOR
    )

    # Check shape
    assert transferred_colors.shape == (4, 3)

    # Check that colors are from source (exact matches expected due to proximity)
    expected_colors = source_pc["label"]
    for i in range(4):
        # Check if transferred color matches any source color
        matches = [torch.allclose(transferred_colors[i], expected_colors[j], atol=1e-6) for j in range(4)]
        assert any(matches), f"Transferred color {i} doesn't match any source color"


def test_transfer_colors_nearest_neighbor_tensors(sample_point_clouds):
    """Test nearest neighbor color transfer with tensor inputs."""
    source_pc, target_pc = sample_point_clouds

    transferred_colors = transfer_colors(
        source_pc["pos"], target_pc["pos"],
        method=ColorTransferMethod.NEAREST_NEIGHBOR,
        source_colors=source_pc["label"]
    )

    assert transferred_colors.shape == (4, 3)


def test_transfer_labels_cpd_weighted(sample_point_clouds):
    """Test CPD-weighted color transfer."""
    source_pc, target_pc = sample_point_clouds

    # Create mock probability matrix (identity for simplicity)
    prob_matrix = torch.eye(4, dtype=torch.float32)
    estep_result = MockEstepResult(prob_matrix)

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.CPD_WEIGHTED,
        estep_result=estep_result,
        pmat_layout="receiver_provider",
    )

    # With identity probabilities, should get exact source colors
    assert torch.allclose(transferred_colors, source_pc["label"], atol=1e-6)


def test_transfer_labels_cpd_weighted_uniform(sample_point_clouds):
    """Test CPD-weighted with uniform probabilities."""
    source_pc, target_pc = sample_point_clouds

    # Uniform probabilities (each target equally likely to match any source)
    prob_matrix = torch.full((4, 4), 0.25, dtype=torch.float32)
    estep_result = MockEstepResult(prob_matrix)

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.CPD_WEIGHTED,
        estep_result=estep_result,
        pmat_layout="receiver_provider",
    )

    # Should get average of all source colors
    expected_avg = source_pc["label"].mean(dim=0)
    assert torch.allclose(transferred_colors, expected_avg, atol=1e-6)


def test_transfer_colors_invalid_method(sample_point_clouds):
    """Test error handling for invalid method."""
    source_pc, target_pc = sample_point_clouds

    with pytest.raises(ValueError, match="Unknown label transfer method"):
        transfer_colors(source_pc, target_pc, method="invalid_method")


def test_transfer_colors_missing_estep_result(sample_point_clouds):
    """Test error handling when estep_result is missing for CPD method."""
    source_pc, target_pc = sample_point_clouds

    with pytest.raises(ValueError, match="estep_result is required"):
        transfer_colors(
            source_pc, target_pc,
            method=ColorTransferMethod.CPD_WEIGHTED
        )


def test_transfer_colors_missing_source_colors():
    """Test error handling when source_colors is missing for tensor input."""
    source_pos = torch.randn(4, 3)
    target_pos = torch.randn(4, 3)

    with pytest.raises(ValueError, match="source_colors must be provided"):
        transfer_colors(
            source_pos, target_pos,
            method=ColorTransferMethod.NEAREST_NEIGHBOR
        )


def test_transfer_colors_dimension_mismatch(sample_point_clouds):
    """Test error handling for mismatched dimensions."""
    source_pc, target_pc = sample_point_clouds

    # Create target with different dimensionality
    target_2d = zRegPointCloud(pos=torch.randn(4, 2))

    with pytest.raises(ValueError, match="same dimensionality"):
        transfer_colors(source_pc, target_2d, method=ColorTransferMethod.NEAREST_NEIGHBOR)


def test_transfer_colors_prob_matrix_transpose(sample_point_clouds):
    """A provider_receiver pmat is transposed internally (declared, never inferred; U6-9)."""
    source_pc, target_pc = sample_point_clouds

    prob_matrix = torch.eye(4, dtype=torch.float32)
    prob_matrix[0, 1] = 0.5  # non-symmetric, so orientation is observable
    estep_result = MockEstepResult(prob_matrix.T)  # (n_source, n_target)

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.CPD_WEIGHTED,
        estep_result=estep_result,
        pmat_layout="provider_receiver",
    )
    expected = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.CPD_WEIGHTED,
        estep_result=MockEstepResult(prob_matrix),
        pmat_layout="receiver_provider",
    )

    assert transferred_colors.shape == (4, 3)
    torch.testing.assert_close(transferred_colors, expected)


@pytest.fixture
def sample_point_clouds_single_channel():
    """Create sample point clouds with single-channel colors (class indices)."""
    # Source point cloud
    source_pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ], dtype=torch.float32)

    source_colors = torch.tensor([
        [0],  # Class 0
        [1],  # Class 1
        [2],  # Class 2
        [0]   # Class 0
    ], dtype=torch.float32)

    source_pc = zRegPointCloud(pos=source_pos, label=source_colors)

    # Target point cloud (slightly offset)
    target_pos = torch.tensor([
        [0.1, 0.1, 0.0],  # Close to source[0] (class 0)
        [0.9, 0.1, 0.0],  # Close to source[1] (class 1)
        [0.1, 0.9, 0.0],  # Close to source[2] (class 2)
        [0.9, 0.9, 0.0]   # Close to source[3] (class 0)
    ], dtype=torch.float32)

    target_pc = zRegPointCloud(pos=target_pos)

    return source_pc, target_pc


def test_transfer_colors_knn_voting(sample_point_clouds_single_channel):
    """Test KNN voting color transfer."""
    source_pc, target_pc = sample_point_clouds_single_channel

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.KNN_VOTING,
        k=1  # With k=1, should be same as nearest neighbor
    )

    assert transferred_colors.shape == (4, 1)

    # With k=1, should match nearest neighbor
    expected_classes = torch.tensor([[0], [1], [2], [0]], dtype=torch.float32)
    assert torch.allclose(transferred_colors, expected_classes, atol=1e-6)


def test_transfer_colors_knn_voting_k3(sample_point_clouds_single_channel):
    """Test KNN voting with k=3."""
    source_pc, target_pc = sample_point_clouds_single_channel

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.KNN_VOTING,
        k=3
    )

    assert transferred_colors.shape == (4, 1)

    # With k=3, each target should get the majority class among its 3 nearest neighbors
    # For this setup, the 3 nearest for each should include itself and neighbors
    # But since positions are close, it might vary, but let's just check shape and range
    assert transferred_colors.min() >= 0
    assert transferred_colors.max() <= 2


def test_transfer_colors_knn_voting_multichannel_error(sample_point_clouds):
    """Test KNN voting raises error for multi-channel colors."""
    source_pc, target_pc = sample_point_clouds  # This has 3-channel colors

    with pytest.raises(ValueError, match="single-channel"):
        transfer_colors(
            source_pc, target_pc,
            method=ColorTransferMethod.KNN_VOTING,
            k=3
        )


def test_transfer_colors_gaussian_kernel(sample_point_clouds):
    """Test Gaussian kernel color transfer."""
    source_pc, target_pc = sample_point_clouds

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.GAUSSIAN_KERNEL,
        sigma=0.1  # Small sigma, should be close to nearest neighbor
    )

    assert transferred_colors.shape == (4, 3)

    # Colors should be weighted averages
    # Check that values are reasonable (between 0 and 1 for RGB-like colors)
    assert transferred_colors.min() >= 0
    assert transferred_colors.max() <= 1


def test_transfer_colors_gaussian_kernel_large_sigma(sample_point_clouds):
    """Test Gaussian kernel with large sigma (should average all colors)."""
    source_pc, target_pc = sample_point_clouds

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.GAUSSIAN_KERNEL,
        sigma=10.0  # Large sigma, uniform weights
    )

    assert transferred_colors.shape == (4, 3)

    # Should be close to the average of all source colors
    expected_avg = source_pc["label"].mean(dim=0)
    assert torch.allclose(transferred_colors, expected_avg, atol=1e-2)

class TestColorTransferEdgeCases:
    """Regression tests for color transfer edge cases (TEST-05)."""

    def test_empty_source_tensor_raises_valueerror(self):
        """transfer_colors raises ValueError with descriptive message for 0-point source (Tensor)."""
        source_pos = torch.zeros(0, 3)
        source_colors = torch.zeros(0, 3)
        target_pos = torch.randn(5, 3)
        with pytest.raises(ValueError, match="0 points"):
            transfer_colors(source_pos, target_pos, source_colors=source_colors)

    def test_empty_source_zreg_raises_valueerror(self):
        """transfer_colors raises ValueError with descriptive message for 0-point source (zRegPointCloud)."""
        source = zRegPointCloud(pos=torch.zeros(0, 3), label=torch.zeros(0, 3))
        target = zRegPointCloud(pos=torch.randn(5, 3))
        with pytest.raises(ValueError, match="0 points"):
            transfer_colors(source, target)

    def test_single_point_source_returns_correct_shape(self):
        """transfer_colors returns (m_target, n_channels) for a single-point source."""
        source = zRegPointCloud(
            pos=torch.tensor([[0.0, 0.0, 0.0]]),
            label=torch.tensor([[0.2, 0.5, 0.8]]),
        )
        target = zRegPointCloud(pos=torch.randn(7, 3))
        result = transfer_colors(source, target)
        assert result.shape == (7, 3)
        assert torch.isfinite(result).all()

    def test_single_point_source_all_targets_get_same_color(self):
        """With a single source point, all target points receive that point's color."""
        color = torch.tensor([[0.1, 0.9, 0.4]])
        source = zRegPointCloud(pos=torch.tensor([[0.0, 0.0, 0.0]]), label=color)
        target = zRegPointCloud(pos=torch.randn(5, 3))
        result = transfer_colors(source, target)
        assert torch.allclose(result, color.expand(5, 3), atol=1e-5)

    def test_dimension_mismatch_raises_valueerror(self):
        """transfer_colors raises ValueError when source and target have different spatial dims."""
        source_pos = torch.randn(10, 3)
        source_colors = torch.zeros(10, 3)
        target_pos = torch.randn(5, 2)
        with pytest.raises(ValueError, match="dimensionality"):
            transfer_colors(source_pos, target_pos, source_colors=source_colors)


def _square_nonsymmetric_pmat():
    return torch.tensor([
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8],
        [0.3, 0.6, 0.1],
    ])


def _three_point_clouds():
    source_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target_pos = source_pos + 0.05
    source_colors = torch.eye(3)
    return source_pos, target_pos, source_colors


def test_cpd_weighted_layouts_agree_on_square_nonsymmetric_pmat():
    """U6-9: receiver_provider P equals provider_receiver P.T for a square non-symmetric pmat."""
    source_pos, target_pos, source_colors = _three_point_clouds()
    P = _square_nonsymmetric_pmat()
    a = transfer_labels(
        source_pos, target_pos, method="cpd_weighted", source_colors=source_colors,
        estep_result=MockEstepResult(P), pmat_layout="receiver_provider",
    )
    b = transfer_labels(
        source_pos, target_pos, method="cpd_weighted", source_colors=source_colors,
        estep_result=MockEstepResult(P.T.contiguous()), pmat_layout="provider_receiver",
    )
    torch.testing.assert_close(a, b)
    # receiver_provider rows are used as is (rows already sum to 1 here)
    torch.testing.assert_close(a, P)


def test_cpd_weighted_missing_pmat_layout_raises():
    """U6-9: cpd_weighted with estep_result but no pmat_layout names both layouts."""
    source_pos, target_pos, source_colors = _three_point_clouds()
    with pytest.raises(ValueError, match="receiver_provider.*provider_receiver"):
        transfer_labels(
            source_pos, target_pos, method="cpd_weighted", source_colors=source_colors,
            estep_result=MockEstepResult(_square_nonsymmetric_pmat()),
        )


def test_cpd_weighted_unknown_pmat_layout_raises():
    source_pos, target_pos, source_colors = _three_point_clouds()
    with pytest.raises(ValueError, match="pmat_layout"):
        transfer_labels(
            source_pos, target_pos, method="cpd_weighted", source_colors=source_colors,
            estep_result=MockEstepResult(_square_nonsymmetric_pmat()), pmat_layout="rows_are_targets",
        )


@pytest.mark.parametrize("layout, shape", [
    ("receiver_provider", (4, 3)),  # this is (n_source, n_target): wrong for receiver_provider
    ("provider_receiver", (3, 4)),  # this is (n_target, n_source): wrong for provider_receiver
    ("receiver_provider", (5, 5)),
])
def test_cpd_weighted_declared_layout_shape_mismatch_raises(layout, shape):
    """U6-9: the declared layout's exact shape is checked; no shape inference."""
    source_pos = torch.randn(4, 3)
    target_pos = torch.randn(3, 3)
    source_colors = torch.randn(4, 3)
    with pytest.raises(ValueError, match="expected"):
        transfer_labels(
            source_pos, target_pos, method="cpd_weighted", source_colors=source_colors,
            estep_result=MockEstepResult(torch.ones(*shape)), pmat_layout=layout,
        )


# ── Gaussian kernel stability (U6-3) ──────────────────────────────────────────

def test_gaussian_kernel_far_target_is_finite_and_nearest():
    """U6-3: every source far from the target -> no 0/0 NaN; the nearest source dominates."""
    source_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source_colors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_pos = torch.tensor([[100.0, 0.0, 0.0]])
    out = transfer_labels(
        source_pos, target_pos, method="gaussian_kernel", source_colors=source_colors, sigma=1.0,
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.tensor([[0.0, 1.0]]))


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_gaussian_kernel_invalid_sigma_raises(sigma):
    source_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source_colors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="sigma"):
        transfer_labels(
            source_pos, source_pos, method="gaussian_kernel", source_colors=source_colors, sigma=sigma,
        )


# ── Source label extraction (U6-5) ────────────────────────────────────────────

def test_zreg_source_without_label_raises():
    """U6-5: a zRegPointCloud without 'label' and no source_colors -> ValueError, not TypeError."""
    source = zRegPointCloud(pos=torch.randn(4, 3))
    target = zRegPointCloud(pos=torch.randn(3, 3))
    with pytest.raises(ValueError, match="label"):
        transfer_labels(source, target)


def test_explicit_source_colors_win_over_zreg_label():
    """U6-5: an explicit source_colors argument is not clobbered by source['label']."""
    source = zRegPointCloud(pos=torch.randn(4, 3), label=torch.zeros(4, 2))
    target = zRegPointCloud(pos=torch.randn(3, 3))
    out = transfer_labels(source, target, source_colors=torch.ones(4, 2))
    torch.testing.assert_close(out, torch.ones(3, 2))


@pytest.mark.parametrize("method, extra", [
    ("nearest_neighbor", {}),
    ("knn_voting", {"k": 2}),
    ("gaussian_kernel", {"sigma": 0.5}),
    ("cpd_weighted", {"pmat_layout": "receiver_provider"}),
])
def test_transfer_does_not_mutate_inputs(method, extra):
    """U6-5: caller tensors (positions, labels, explicit source_colors) are never written to."""
    gen = torch.Generator().manual_seed(0)
    src_pos = torch.randn(5, 3, generator=gen)
    tgt_pos = torch.randn(4, 3, generator=gen)
    src_label = torch.tensor([[0.0], [1.0], [2.0], [1.0], [0.0]])
    explicit = torch.tensor([[1.0], [0.0], [2.0], [2.0], [1.0]])
    source = zRegPointCloud(pos=src_pos, label=src_label)
    target = zRegPointCloud(pos=tgt_pos)
    originals = [t.clone() for t in (src_pos, src_label, tgt_pos, explicit)]
    kwargs = dict(extra)
    if method == "cpd_weighted":
        pmat = torch.rand(4, 5, generator=gen) + 0.1
        pmat_orig = pmat.clone()
        kwargs["estep_result"] = MockEstepResult(pmat)
    transfer_labels(source, target, method=method, source_colors=explicit, **kwargs)
    for orig, now in zip(originals, (src_pos, src_label, tgt_pos, explicit)):
        assert torch.equal(orig, now)
    if method == "cpd_weighted":
        assert torch.equal(pmat_orig, pmat)


# ── Model-based dispatch ──────────────────────────────────────────────────────

def test_egnn_enum_value():
    assert LabelTransferMethod.EGNN.value == "egnn"


def test_pointnet2_enum_value():
    assert LabelTransferMethod.POINTNET2.value == "pointnet2"


def test_egnn_requires_model(sample_point_clouds):
    source, target = sample_point_clouds
    with pytest.raises(ValueError, match="model is required"):
        transfer_labels(source, target, method="egnn")


def test_pointnet2_requires_model(sample_point_clouds):
    source, target = sample_point_clouds
    with pytest.raises(ValueError, match="model is required"):
        transfer_labels(source, target, method="pointnet2")


def _tiny_egnn():
    from zreg.models.egnn import EGNNLabelTransfer

    torch.manual_seed(0)
    return EGNNLabelTransfer(n_classes=3, hidden_dim=8, n_layers=1)


def _tiny_pointnet2():
    from zreg.models.pointnet2 import PointNet2LabelTransfer

    torch.manual_seed(0)
    return PointNet2LabelTransfer(n_classes=3, hidden_dim=8)


_TINY_MODELS = [("egnn", _tiny_egnn), ("pointnet2", _tiny_pointnet2)]


def _model_clouds():
    gen = torch.Generator().manual_seed(1)
    source_pos = torch.rand(24, 3, generator=gen)
    target_pos = torch.rand(16, 3, generator=gen)
    labels = torch.arange(24) % 3
    return source_pos, target_pos, labels


@pytest.mark.parametrize("method, factory", _TINY_MODELS)
def test_transfer_labels_model_accepts_1d_labels(method, factory):
    """U6-1: 1-D integer labels are one-hot encoded with model.n_classes (real tiny model)."""
    source_pos, target_pos, labels = _model_clouds()
    model = factory()
    torch.manual_seed(2)
    out = transfer_labels(source_pos, target_pos, method=method, source_colors=labels, model=model)
    assert out.shape == (16, 3)
    assert bool(torch.isfinite(out).all())
    torch.testing.assert_close(out.sum(dim=1), torch.ones(16), atol=1e-5, rtol=0)


@pytest.mark.parametrize("method, factory", _TINY_MODELS)
def test_transfer_labels_model_one_hot_matches_1d(method, factory):
    source_pos, target_pos, labels = _model_clouds()
    model = factory()
    torch.manual_seed(2)
    from_ids = transfer_labels(source_pos, target_pos, method=method, source_colors=labels, model=model)
    torch.manual_seed(2)
    from_one_hot = transfer_labels(
        source_pos, target_pos, method=method,
        source_colors=torch.nn.functional.one_hot(labels, num_classes=3).float(), model=model,
    )
    torch.testing.assert_close(from_ids, from_one_hot)


@pytest.mark.parametrize("method, factory", _TINY_MODELS)
def test_transfer_labels_model_width_mismatch_raises(method, factory):
    source_pos, target_pos, labels = _model_clouds()
    with pytest.raises(ValueError, match="n_classes"):
        transfer_labels(
            source_pos, target_pos, method=method,
            source_colors=torch.nn.functional.one_hot(labels % 2, num_classes=2).float(),
            model=factory(),
        )


@pytest.mark.parametrize("method, factory", _TINY_MODELS)
def test_transfer_labels_model_out_of_range_label_raises(method, factory):
    source_pos, target_pos, labels = _model_clouds()
    labels = labels.clone()
    labels[0] = 3
    with pytest.raises(ValueError, match="n_classes"):
        transfer_labels(source_pos, target_pos, method=method, source_colors=labels, model=factory())


def test_transfer_labels_model_without_n_classes_raises():
    class _NoClasses(torch.nn.Module):
        def forward(self, joint_pos, joint_feat):
            return torch.zeros(joint_pos.shape[0], 3)

    source_pos, target_pos, labels = _model_clouds()
    with pytest.raises(ValueError, match="n_classes"):
        transfer_labels(source_pos, target_pos, method="egnn", source_colors=labels, model=_NoClasses())


class _RecordingModel(torch.nn.Module):
    """A real module that records its input features and returns zero logits."""

    n_classes = 3

    def __init__(self):
        super().__init__()
        self.joint_feat = None

    def forward(self, joint_pos, joint_feat):
        self.joint_feat = joint_feat.clone()
        return torch.zeros(joint_pos.shape[0], self.n_classes)


def test_transfer_labels_model_joint_feat_encoding():
    """Joint features: source rows one-hot + unknown_flag 0, target rows zeros + unknown_flag 1."""
    source_pos, target_pos, labels = _model_clouds()
    model = _RecordingModel()
    out = transfer_labels(source_pos, target_pos, method="egnn", source_colors=labels, model=model)
    jf = model.joint_feat
    n_source = source_pos.shape[0]
    assert jf.shape == (n_source + target_pos.shape[0], 4)
    torch.testing.assert_close(jf[:n_source, :3], torch.nn.functional.one_hot(labels, 3).float())
    assert (jf[:n_source, 3] == 0.0).all()
    assert (jf[n_source:, :3] == 0.0).all()
    assert (jf[n_source:, 3] == 1.0).all()
    torch.testing.assert_close(out, torch.full((target_pos.shape[0], 3), 1.0 / 3.0))


# ── Shared pmat zero-row policy: repair_pmat_rows (U6-7, IN-09a, IN-09c) ──────

import warnings  # noqa: E402

from zreg.label_transfer import (  # noqa: E402
    MAX_PMAT_FALLBACK_FRACTION,
    PmatRowRepair,
    repair_pmat_rows,
)

_PROVIDER4 = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
])
_RECEIVER4 = torch.tensor([
    [0.1, 0.1, 0.0],
    [0.9, 0.1, 0.0],
    [0.1, 0.9, 0.0],
    [0.9, 0.9, 0.0],
])
_COLOURS4 = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])


def _uniform_pmat():
    return torch.full((4, 4), 0.25)


def _posterior_warnings(records):
    return [
        w for w in records
        if issubclass(w.category, RuntimeWarning) and "posterior mass" in str(w.message)
    ]


def test_repair_pmat_rows_zero_row_falls_back_to_nearest_provider():
    pmat = _uniform_pmat()
    pmat[1] = 0.0
    pmat_orig = pmat.clone()
    repair = repair_pmat_rows(pmat, _PROVIDER4, _RECEIVER4, context="frame 3")
    assert isinstance(repair, PmatRowRepair)
    assert repair.bad_rows.tolist() == [False, True, False, False]
    nearest = torch.cdist(_RECEIVER4[1:2], _PROVIDER4, p=2).argmin().item()
    assert repair.fallback_idx.tolist() == [nearest]
    assert repair.n_bad == 1 and repair.n_nonfinite_pos == 0
    mass = repair.pmat.sum(dim=1)
    assert bool(torch.isfinite(mass).all()) and bool((mass > 0).all())
    assert torch.equal(pmat, pmat_orig)
    assert "1 of 4" in repair.message and "frame 3" in repair.message


def test_repair_pmat_rows_nan_row_treated_as_zero_row():
    pmat = _uniform_pmat()
    pmat[2, 1] = float("nan")
    repair = repair_pmat_rows(pmat, _PROVIDER4, _RECEIVER4, context="frame 3")
    assert repair.bad_rows.tolist() == [False, False, True, False]
    assert repair.fallback_idx.tolist() == [2]
    assert bool(torch.isfinite(repair.pmat).all())


def test_repair_pmat_rows_nonfinite_receiver_position_gets_minus_one():
    """IN-09a: a NaN receiver position is a bad row even with positive mass; no argmin over NaN."""
    receiver = _RECEIVER4.clone()
    receiver[0, 0] = float("nan")
    repair = repair_pmat_rows(_uniform_pmat(), _PROVIDER4, receiver, context="frame 3")
    assert repair.bad_rows.tolist() == [True, False, False, False]
    assert repair.fallback_idx.tolist() == [-1]
    assert repair.n_nonfinite_pos == 1
    assert "(-1)" in repair.message


def test_repair_pmat_rows_never_picks_nonfinite_provider():
    provider = _PROVIDER4.clone()
    provider[1] = float("nan")  # receiver 1's true nearest is provider 1
    pmat = _uniform_pmat()
    pmat[1] = 0.0
    repair = repair_pmat_rows(pmat, provider, _RECEIVER4, context="frame 3")
    assert repair.fallback_idx.item() != 1
    finite = provider.clone()
    finite[1] = 1e9
    assert repair.fallback_idx.item() == torch.cdist(_RECEIVER4[1:2], finite).argmin().item()


def test_repair_pmat_rows_no_finite_provider_raises():
    provider = torch.full((4, 3), float("nan"))
    pmat = _uniform_pmat()
    pmat[1] = 0.0
    with pytest.raises(ValueError, match="finite"):
        repair_pmat_rows(pmat, provider, _RECEIVER4, context="frame 3")


def test_repair_pmat_rows_validates_pmat_1d():
    with pytest.raises(ValueError, match="2-D"):
        repair_pmat_rows(torch.ones(4), _PROVIDER4, _RECEIVER4)


def test_repair_pmat_rows_validates_pmat_3d():
    with pytest.raises(ValueError, match="2-D"):
        repair_pmat_rows(torch.ones(4, 4, 1), _PROVIDER4, _RECEIVER4)


def test_repair_pmat_rows_validates_shape():
    with pytest.raises(ValueError, match=r"shape.*\(4, 3\).*\(4, 4\)"):
        repair_pmat_rows(torch.ones(4, 3), _PROVIDER4, _RECEIVER4)


def test_repair_pmat_rows_validates_coordinate_width():
    with pytest.raises(ValueError, match="coordinate"):
        repair_pmat_rows(torch.ones(4, 4), _PROVIDER4, _RECEIVER4[:, :2])


def test_repair_pmat_rows_validates_device():
    pmat = torch.ones(4, 4, device=torch.device("meta"))
    with pytest.raises(ValueError, match="device"):
        repair_pmat_rows(pmat, _PROVIDER4, _RECEIVER4)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan")])
def test_repair_pmat_rows_validates_max_fallback_fraction(bad):
    with pytest.raises(ValueError, match="max_fallback_fraction"):
        repair_pmat_rows(_uniform_pmat(), _PROVIDER4, _RECEIVER4, max_fallback_fraction=bad)


def test_cpd_weighted_validates_source_colors_rows():
    with pytest.raises(ValueError, match="source_colors"):
        transfer_labels(
            _PROVIDER4, _RECEIVER4, method="cpd_weighted", source_colors=_COLOURS4[:3],
            estep_result=MockEstepResult(_uniform_pmat()), pmat_layout="receiver_provider",
        )


def test_repair_pmat_rows_zero_receivers_returns_empty():
    repair = repair_pmat_rows(torch.zeros(0, 4), _PROVIDER4, torch.zeros(0, 3), context="frame 3")
    assert repair.n_bad == 0 and repair.n_nonfinite_pos == 0
    assert repair.message == ""
    assert repair.bad_rows.shape == (0,)
    assert repair.fallback_idx.shape == (0,)
    assert repair.pmat.shape == (0, 4)


def test_cpd_weighted_empty_target_returns_empty():
    """Pins existing behaviour (GREEN at HEAD by design): empty target -> (0, n_channels), no warning."""
    source = zRegPointCloud(pos=_PROVIDER4, label=_COLOURS4[:, :2])
    target = zRegPointCloud(pos=torch.zeros(0, 3))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = transfer_labels(
            source, target, method="cpd_weighted",
            estep_result=MockEstepResult(torch.zeros(0, 4)), pmat_layout="receiver_provider",
        )
    assert out.shape == (0, 2)
    assert rec == []


def test_repair_pmat_rows_zero_providers_raises_every_receiver():
    with pytest.raises(ValueError, match="every receiver point"):
        repair_pmat_rows(torch.zeros(4, 0), torch.zeros(0, 3), _RECEIVER4, context="frame 3")


def _cpd(pmat, receiver=_RECEIVER4, **kwargs):
    return transfer_labels(
        _PROVIDER4, receiver, method="cpd_weighted", source_colors=_COLOURS4,
        estep_result=MockEstepResult(pmat), pmat_layout="receiver_provider", **kwargs,
    )


def test_cpd_weighted_zero_row_falls_back_with_warning():
    """U6-7: a zero-mass row is not NaN; it takes the nearest provider's colour row."""
    pmat = _uniform_pmat()
    pmat[1] = 0.0
    with pytest.warns(RuntimeWarning, match="zero or non-finite posterior mass"):
        out = _cpd(pmat)
    assert bool(torch.isfinite(out).all())
    torch.testing.assert_close(out[1], _COLOURS4[1])
    torch.testing.assert_close(out[0], _COLOURS4.mean(dim=0))


def test_cpd_weighted_nan_row_falls_back_with_warning():
    pmat = _uniform_pmat()
    pmat[3, 0] = float("nan")
    with pytest.warns(RuntimeWarning, match="zero or non-finite posterior mass"):
        out = _cpd(pmat)
    assert bool(torch.isfinite(out).all())
    torch.testing.assert_close(out[3], _COLOURS4[3])


def test_cpd_weighted_all_rows_zero_raises():
    with pytest.raises(ValueError, match="every receiver point"):
        _cpd(torch.zeros(4, 4))


def test_cpd_weighted_fallback_fraction_above_bound_raises():
    """IN-09c: 3 of 4 rows (75%) > 0.5 -> raise naming the bound and the library context."""
    pmat = _uniform_pmat()
    pmat[:3] = 0.0
    with pytest.raises(ValueError, match=r"max_fallback_fraction=0\.5") as exc:
        _cpd(pmat)
    assert "in cpd_weighted pmat" in str(exc.value)
    assert MAX_PMAT_FALLBACK_FRACTION == 0.5


def test_cpd_weighted_fallback_fraction_exactly_half_falls_back():
    pmat = _uniform_pmat()
    pmat[:2] = 0.0
    with pytest.warns(RuntimeWarning, match="2 of 4"):
        out = _cpd(pmat)
    torch.testing.assert_close(out[:2], _COLOURS4[:2])


def test_cpd_weighted_custom_max_fallback_fraction():
    pmat = _uniform_pmat()
    pmat[:3] = 0.0
    with pytest.warns(RuntimeWarning):
        out = _cpd(pmat, max_fallback_fraction=0.8)
    torch.testing.assert_close(out[:3], _COLOURS4[:3])


@pytest.mark.parametrize("bad", [0, 1.5])
def test_cpd_weighted_invalid_max_fallback_fraction_raises(bad):
    with pytest.raises(ValueError, match="max_fallback_fraction"):
        _cpd(_uniform_pmat(), max_fallback_fraction=bad)


def test_cpd_weighted_nonfinite_receiver_position_gives_zero_row():
    """IN-09a (library): a NaN receiver position gets an all-zero ('no label') row, others unaffected."""
    receiver = _RECEIVER4.clone()
    receiver[0] = torch.tensor([float("nan"), 0.0, 0.0])
    pmat = _uniform_pmat()
    pmat[0] = float("nan")
    with pytest.warns(RuntimeWarning, match="zero or non-finite posterior mass"):
        out = _cpd(pmat, receiver=receiver)
    assert bool(torch.isfinite(out).all())
    assert torch.equal(out[0], torch.zeros(3))
    torch.testing.assert_close(out[1:], _COLOURS4.mean(dim=0).expand(3, 3))


def test_cpd_weighted_pmat_repair_reused_without_second_warning():
    """RD-1b: a precomputed repair is reused as is -- no second repair, no warning."""
    pmat = _uniform_pmat()
    pmat[1] = 0.0
    repair = repair_pmat_rows(pmat, _PROVIDER4, _RECEIVER4, context="frame 3")
    with warnings.catch_warnings(record=True) as rec_reuse:
        warnings.simplefilter("always")
        reused = transfer_labels(
            _PROVIDER4, _RECEIVER4, method="cpd_weighted", source_colors=_COLOURS4,
            pmat_repair=repair,
        )
    with warnings.catch_warnings(record=True) as rec_self:
        warnings.simplefilter("always")
        self_repaired = _cpd(pmat)
    assert _posterior_warnings(rec_reuse) == []
    assert len(_posterior_warnings(rec_self)) == 1
    torch.testing.assert_close(reused, self_repaired)


def test_cpd_weighted_pmat_repair_nonfinite_position_no_warning():
    receiver = _RECEIVER4.clone()
    receiver[0, 1] = float("nan")
    repair = repair_pmat_rows(_uniform_pmat(), _PROVIDER4, receiver, context="frame 3")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = transfer_labels(
            _PROVIDER4, receiver, method="cpd_weighted", source_colors=_COLOURS4,
            pmat_repair=repair,
        )
    assert _posterior_warnings(rec) == []
    assert torch.equal(out[0], torch.zeros(3))


def test_cpd_weighted_pmat_repair_and_estep_result_raises():
    repair = repair_pmat_rows(_uniform_pmat(), _PROVIDER4, _RECEIVER4)
    with pytest.raises(ValueError, match="pmat_repair"):
        transfer_labels(
            _PROVIDER4, _RECEIVER4, method="cpd_weighted", source_colors=_COLOURS4,
            estep_result=MockEstepResult(_uniform_pmat()), pmat_layout="receiver_provider",
            pmat_repair=repair,
        )


def test_cpd_weighted_pmat_repair_neither_given_raises():
    with pytest.raises(ValueError, match="estep_result"):
        transfer_labels(_PROVIDER4, _RECEIVER4, method="cpd_weighted", source_colors=_COLOURS4)


def test_cpd_weighted_pmat_repair_shape_mismatch_raises():
    repair = repair_pmat_rows(torch.full((3, 4), 0.25), _PROVIDER4, _RECEIVER4[:3])
    with pytest.raises(ValueError, match="shape"):
        transfer_labels(
            _PROVIDER4, _RECEIVER4, method="cpd_weighted", source_colors=_COLOURS4,
            pmat_repair=repair,
        )


def test_label_transfer_all_exports_repair_api():
    import zreg.label_transfer as lt

    assert {
        "repair_pmat_rows", "PmatRowRepair", "MAX_PMAT_FALLBACK_FRACTION",
        "transfer_labels", "LabelTransferMethod",
    } <= set(lt.__all__)
    namespace: dict = {}
    exec("from zreg.label_transfer import *", namespace)
    assert "repair_pmat_rows" in namespace
