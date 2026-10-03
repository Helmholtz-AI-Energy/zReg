"""Tests for color transfer functionality."""

import unittest.mock
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


def test_transfer_labels_egnn_mock(sample_point_clouds):
    """Dispatch calls model.forward() and returns (n_target, n_classes) softmax probs."""
    source, target = sample_point_clouds
    n_target = target["pos"].shape[0]
    n_classes = source["label"].shape[1]

    mock_model = unittest.mock.MagicMock()
    fake_logits = torch.zeros(source["pos"].shape[0] + n_target, n_classes)
    mock_model.return_value = fake_logits

    result = transfer_labels(source, target, method="egnn", model=mock_model)

    assert result.shape == (n_target, n_classes)
    mock_model.assert_called_once()
    mock_model.eval.assert_called_once()


def test_transfer_labels_pointnet2_mock(sample_point_clouds):
    """PointNet2 path mirrors EGNN path (same _transfer_labels_model helper)."""
    source, target = sample_point_clouds
    n_target = target["pos"].shape[0]
    n_classes = source["label"].shape[1]

    mock_model = unittest.mock.MagicMock()
    fake_logits = torch.zeros(source["pos"].shape[0] + n_target, n_classes)
    mock_model.return_value = fake_logits

    result = transfer_labels(source, target, method="pointnet2", model=mock_model)

    assert result.shape == (n_target, n_classes)


def test_transfer_labels_model_joint_feat_encoding(sample_point_clouds):
    """Joint feature matrix: source rows have unknown_flag=0, target rows have unknown_flag=1."""
    source, target = sample_point_clouds
    n_source = source["pos"].shape[0]
    n_classes = source["label"].shape[1]
    captured = {}

    def capture_forward(joint_pos, joint_feat):
        captured["joint_feat"] = joint_feat
        return torch.zeros(joint_pos.shape[0], n_classes)

    mock_model = unittest.mock.MagicMock()
    mock_model.side_effect = capture_forward

    transfer_labels(source, target, method="egnn", model=mock_model)

    jf = captured["joint_feat"]
    assert (jf[:n_source, n_classes] == 0.0).all()
    assert (jf[n_source:, n_classes] == 1.0).all()
    torch.testing.assert_close(jf[:n_source, :n_classes], source["label"].float())

