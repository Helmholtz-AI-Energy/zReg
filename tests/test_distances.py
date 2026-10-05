"""Tests for zreg.distance_metrics module."""

import pytest
import torch

from zreg import distance_metrics as distances
from zreg.distance_metrics import general
from zreg.distance_metrics import sw_varients


@pytest.fixture
def point_clouds_2d():
    """Create 2D point cloud pair."""
    x = torch.randn(100, 2)
    y = torch.randn(100, 2)
    return x, y


@pytest.fixture
def point_clouds_3d():
    """Create 3D point cloud pair."""
    x = torch.randn(100, 3)
    y = torch.randn(100, 3)
    return x, y


@pytest.fixture
def batched_point_clouds():
    """Create batched point clouds for SWD variants."""
    x = torch.randn(2, 100, 3)  # batch_size=2, 100 points, 3D
    y = torch.randn(2, 100, 3)
    return x, y


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_shape(self, point_clouds_3d):
        """Test output shape."""
        x, y = point_clouds_3d
        result = general.euclidean_distance(x, y)
        assert result.shape == (100, 100)

    def test_zero_distance_same_points(self):
        """Test that distance to self is zero."""
        x = torch.randn(10, 3)
        result = general.euclidean_distance(x, x)
        diagonal = torch.diag(result)
        assert torch.allclose(diagonal, torch.zeros(10), atol=1e-5)

    def test_positive_values(self, point_clouds_3d):
        """Test that distances are non-negative."""
        x, y = point_clouds_3d
        result = general.euclidean_distance(x, y)
        assert (result >= 0).all()

    def test_with_normalization(self, point_clouds_3d):
        """Test with normalization enabled."""
        x, y = point_clouds_3d
        result = general.euclidean_distance(x, y, normalize=True)
        assert result.shape == (100, 100)


class TestManhattanDistance:
    """Tests for manhattan_distance function."""

    def test_shape(self, point_clouds_3d):
        """Test output shape."""
        x, y = point_clouds_3d
        result = general.manhattan_distance(x, y)
        assert result.shape == (100, 100)

    def test_positive_values(self, point_clouds_3d):
        """Test that distances are non-negative."""
        x, y = point_clouds_3d
        result = general.manhattan_distance(x, y)
        assert (result >= 0).all()


class TestMinkowskiDistance:
    """Tests for minkowski_distance function."""

    def test_shape(self, point_clouds_3d):
        """Test output shape."""
        x, y = point_clouds_3d
        result = general.minkowski_distance(x, y, p=3)
        assert result.shape == (100, 100)

    def test_p2_equals_euclidean(self, point_clouds_3d):
        """Test that p=2 gives same result as euclidean."""
        x, y = point_clouds_3d
        euclidean = general.euclidean_distance(x, y)
        minkowski_2 = general.minkowski_distance(x, y, p=2)
        assert torch.allclose(euclidean, minkowski_2, atol=1e-5)

    def test_different_p_values(self, point_clouds_3d):
        """Test with different p values."""
        x, y = point_clouds_3d
        for p in [1, 2, 3, 4]:
            result = general.minkowski_distance(x, y, p=p)
            assert result.shape == (100, 100)
            assert (result >= 0).all()


class TestSlicedWassersteinDistance:
    """Tests for SlicedWassersteinDistance class."""

    def test_initialization(self):
        """Test SWD initialization."""
        swd = sw_varients.SlicedWassersteinDistance(num_projs=50)
        assert swd.num_projs == 50

    def test_forward_unbatched(self, point_clouds_3d):
        """Test forward pass with unbatched input."""
        x, y = point_clouds_3d
        swd = sw_varients.SlicedWassersteinDistance(num_projs=50, device="cpu")
        result = swd(x, y)
        assert result.ndim == 0  # scalar

    def test_forward_batched(self, batched_point_clouds):
        """Test forward pass with batched input."""
        x, y = batched_point_clouds
        swd = sw_varients.SlicedWassersteinDistance(
            num_projs=50, device="cpu", nobatchdim=False
        )
        result = swd(x, y)
        assert result.ndim == 0

    def test_result_positive(self, point_clouds_3d):
        """Test that SWD is non-negative."""
        x, y = point_clouds_3d
        swd = sw_varients.SlicedWassersteinDistance(num_projs=50, device="cpu")
        result = swd(x, y)
        assert result >= 0

    def test_same_points_small_distance(self):
        """Test that same points give small distance."""
        x = torch.randn(50, 3)
        swd = sw_varients.SlicedWassersteinDistance(num_projs=100, device="cpu")
        result = swd(x, x)
        assert result < 0.1  # Should be very small


class TestAdaptiveSlicedWassersteinDistance:
    """Tests for AdaptiveSlicedWassersteinDistance class."""

    def test_initialization(self):
        """Test ASWD initialization."""
        aswd = sw_varients.AdaptiveSlicedWassersteinDistance(
            init_projs=20, step_projs=10, max_slices=100
        )
        assert aswd.init_projs == 20
        assert aswd.step_projs == 10
        assert aswd.max_slices == 100

    def test_forward(self, point_clouds_3d):
        """Test forward pass."""
        x, y = point_clouds_3d
        aswd = sw_varients.AdaptiveSlicedWassersteinDistance(
            init_projs=20, step_projs=10, max_slices=50, device="cpu"
        )
        result = aswd(x, y)
        assert result.ndim == 0
        assert result >= 0

    def test_remove_history(self, point_clouds_3d, tmp_path):
        """Test history file removal."""
        import os
        
        x, y = point_clouds_3d
        history_file = str(tmp_path / "test_history.txt")
        
        aswd = sw_varients.AdaptiveSlicedWassersteinDistance(
            init_projs=20, step_projs=10, max_slices=30,
            projs_history=history_file, device="cpu"
        )
        
        # Run to create history file
        aswd(x, y)
        
        # Remove history
        aswd.remove_history()


class TestOrthogonalSlicedWassersteinDistance:
    """Tests for OrthogonalSlicedWassersteinDistance class."""

    def test_initialization(self):
        """Test OSWD initialization."""
        oswd = sw_varients.OrthogonalSlicedWassersteinDistance(num_projs=50)
        assert oswd.num_projs == 50

    def test_forward(self, point_clouds_3d):
        """Test forward pass."""
        x, y = point_clouds_3d
        # OSWD requires num_projs <= dim; point_clouds_3d has dim=3 so cap at 3
        oswd = sw_varients.OrthogonalSlicedWassersteinDistance(
            num_projs=3, device="cpu"
        )
        result = oswd(x, y)
        assert result.ndim == 0
        assert result >= 0


class TestGeneralisedSlicedWassersteinDistance:
    """Tests for GeneralisedSlicedWassersteinDistance class."""

    def test_initialization(self):
        """Test GSWD initialization."""
        gswd = sw_varients.GeneralisedSlicedWassersteinDistance(
            num_projs=50, degree=2.0, g_type="circular"
        )
        assert gswd.num_projs == 50
        assert gswd.degree == 2.0
        assert gswd.g_type == "circular"

    def test_forward_circular(self, point_clouds_3d):
        """Test forward pass with circular projection."""
        x, y = point_clouds_3d
        gswd = sw_varients.GeneralisedSlicedWassersteinDistance(
            num_projs=50, g_type="circular", device="cpu"
        )
        result = gswd(x, y)
        assert result.ndim == 0
        assert result >= 0

    def test_forward_linear(self, point_clouds_3d):
        """Test forward pass with linear projection."""
        x, y = point_clouds_3d
        gswd = sw_varients.GeneralisedSlicedWassersteinDistance(
            num_projs=50, g_type="linear", device="cpu"
        )
        result = gswd(x, y)
        assert result.ndim == 0
        assert result >= 0

    def test_invalid_g_type(self, point_clouds_3d):
        """Test that invalid g_type raises error."""
        x, y = point_clouds_3d
        gswd = sw_varients.GeneralisedSlicedWassersteinDistance(
            num_projs=50, g_type="invalid", device="cpu"
        )
        with pytest.raises(NotImplementedError):
            gswd(x, y)

    def test_default_degree_direct(self):
        """Test GSWD with default degree (no explicit kwarg) returns finite positive distance."""
        torch.manual_seed(42)
        x = torch.randn(1, 30, 3)
        torch.manual_seed(123)
        y = torch.randn(1, 30, 3)
        gswd = sw_varients.GeneralisedSlicedWassersteinDistance(
            num_projs=10, device="cpu", nobatchdim=False
        )
        dist = gswd(x, y)
        assert torch.isfinite(dist).all(), f"GSWD returned non-finite: {dist}"
        assert dist.item() > 0, f"GSWD returned non-positive: {dist}"

    def test_default_degree_via_pairwise_defaults(self):
        """Test that pairwise_distance_matrix GSWD defaults include degree=2.0.

        This verifies the defaults list in _sanitize_pairwise_distance_matrix
        includes degree so that GSWD constructed via dist='gswd' path gets
        the degree parameter even when not explicitly provided by the user.
        """
        from unittest.mock import MagicMock, patch
        import zreg.algorithms.pairwise_distance_matrix as pdm

        # Create minimal mock point clouds with device attribute
        mock_device = torch.device("cpu")
        mock_pos = MagicMock()
        mock_pos.device = mock_device
        mock_pos.dtype = torch.float32
        x = {0: {"pos": mock_pos}}
        y = {0: {"pos": mock_pos}}

        # Capture kwargs passed to GeneralisedSlicedWassersteinDistance constructor
        captured_kwargs = {}
        original_init = sw_varients.GeneralisedSlicedWassersteinDistance.__init__

        def capturing_init(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            original_init(self, *args, **kwargs)

        with patch.object(
            sw_varients.GeneralisedSlicedWassersteinDistance,
            "__init__",
            capturing_init,
        ):
            pdm._sanitize_pairwise_distance_matrix(
                distance_kwargs=[None],
                distance_metrics=["gswd"],
                downsample_method="random",
                x=x,
                y=y,
            )

        assert "degree" in captured_kwargs, (
            f"degree not passed to GSWD constructor via pairwise defaults. "
            f"Got kwargs: {captured_kwargs}"
        )


class TestProjectedWassersteinDistance:
    """Tests for ProjectedWassersteinDistance class."""

    def test_initialization(self):
        """Test PWD initialization."""
        pwd = sw_varients.ProjectedWassersteinDistance(num_projs=50)
        assert pwd.num_projs == 50

    def test_forward(self, point_clouds_3d):
        """Test forward pass."""
        x, y = point_clouds_3d
        pwd = sw_varients.ProjectedWassersteinDistance(
            num_projs=50, device="cpu"
        )
        result = pwd(x, y)
        assert result.ndim == 0
        assert result >= 0

    def test_orthogonal_projections(self, point_clouds_3d):
        """Test with orthogonal projections."""
        x, y = point_clouds_3d
        pwd = sw_varients.ProjectedWassersteinDistance(
            num_projs=50, orthogonal=True, device="cpu"
        )
        result = pwd(x, y)
        assert result.ndim == 0


class TestMaxSlicedWassersteinDistance:
    """Tests for MaxSlicedWassersteinDistance class."""

    def test_initialization(self):
        """Test MaxSWD initialization."""
        max_swd = sw_varients.MaxSlicedWassersteinDistance(device="cpu")
        assert max_swd is not None

    def test_nonidentical_returns_finite_positive(self):
        """Test MaxSWD returns finite positive distance for non-identical clouds."""
        torch.manual_seed(42)
        x = torch.randn(1, 50, 3)
        torch.manual_seed(123)
        y = torch.randn(1, 50, 3)
        mswd = sw_varients.MaxSlicedWassersteinDistance(device="cpu", nobatchdim=False)
        dist = mswd(x, y)
        assert torch.isfinite(dist).all(), f"MaxSWD returned non-finite: {dist}"
        assert dist.item() > 0, f"MaxSWD returned non-positive: {dist}"

    def test_identical_returns_near_zero(self):
        """Test MaxSWD returns near-zero distance for identical clouds."""
        torch.manual_seed(42)
        x = torch.randn(1, 50, 3)
        mswd = sw_varients.MaxSlicedWassersteinDistance(device="cpu", nobatchdim=False)
        dist = mswd(x, x)
        assert dist.item() < 1e-4, f"MaxSWD on identical clouds too large: {dist}"

    def test_no_print_output(self, capsys):
        """Test MaxSWD produces no stdout output (debug prints removed)."""
        torch.manual_seed(42)
        x = torch.randn(1, 50, 3)
        torch.manual_seed(123)
        y = torch.randn(1, 50, 3)
        mswd = sw_varients.MaxSlicedWassersteinDistance(device="cpu", nobatchdim=False)
        mswd(x, y)
        captured = capsys.readouterr()
        assert captured.out == "", f"MaxSWD produced unexpected output: {captured.out[:200]}"


class TestHelperFunctions:
    """Tests for helper functions in sw_varients."""

    def test_minibatch_rand_projections_shape(self):
        """Test minibatch_rand_projections output shape."""
        projs = sw_varients.minibatch_rand_projections(
            batchsize=2, dim=3, num_projections=50
        )
        assert projs.shape == (2, 50, 3)

    def test_minibatch_rand_projections_normalized(self):
        """Test that projections are normalized."""
        projs = sw_varients.minibatch_rand_projections(
            batchsize=2, dim=3, num_projections=50
        )
        norms = torch.sqrt(torch.sum(projs**2, dim=2))
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_proj_onto_unit_sphere(self):
        """Test projection onto unit sphere."""
        vectors = torch.randn(2, 10, 3) * 5  # Non-unit vectors
        projected = sw_varients.proj_onto_unit_sphere(vectors)
        norms = torch.sqrt(torch.sum(projected**2, dim=2))
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_compute_practical_moments_sw(self, batched_point_clouds):
        """Test compute_practical_moments_sw function."""
        x, y = batched_point_clouds
        first_moment, second_moment = sw_varients.compute_practical_moments_sw(
            x, y, num_projections=30
        )
        assert first_moment.shape == (2,)  # batch_size
        assert second_moment.shape == (2,)


class TestDistancesModuleExports:
    """Tests for distances module exports."""

    def test_general_exports(self):
        """Test general distance exports."""
        assert "euclidean_distance" in general.__all__
        assert "manhattan_distance" in general.__all__
        assert "minkowski_distance" in general.__all__

    def test_sw_exports(self):
        """Test Sliced Wasserstein distance exports."""
        assert "SlicedWassersteinDistance" in sw_varients.__all__
        assert "AdaptiveSlicedWassersteinDistance" in sw_varients.__all__
        assert "OrthogonalSlicedWassersteinDistance" in sw_varients.__all__
        assert "GeneralisedSlicedWassersteinDistance" in sw_varients.__all__
        assert "ProjectedWassersteinDistance" in sw_varients.__all__


class TestDistanceConsistency:
    """Tests for consistency between different distance metrics."""

    def test_same_points_zero_or_small(self, point_clouds_3d):
        """Test that all distances are zero or small for same points."""
        x, _ = point_clouds_3d
        x_subset = x[:50]  # Use subset for speed
        
        # Euclidean on same points should have zero diagonal
        # Note: torch.cdist can have small numerical errors due to floating-point precision
        euclidean = general.euclidean_distance(x_subset, x_subset)
        assert torch.allclose(torch.diag(euclidean), torch.zeros(50), atol=2e-3)
        
        # SWD on same points should be small
        swd = sw_varients.SlicedWassersteinDistance(num_projs=100, device="cpu")
        swd_result = swd(x_subset, x_subset)
        assert swd_result < 0.1

    def test_different_dtypes(self):
        """Test distance computation with different dtypes."""
        x = torch.randn(50, 3, dtype=torch.float32)
        y = torch.randn(50, 3, dtype=torch.float32)
        
        # Should work with float32
        result32 = general.euclidean_distance(x, y)
        assert result32.dtype == torch.float32
        
        # Should work with float64
        x64 = x.double()
        y64 = y.double()
        result64 = general.euclidean_distance(x64, y64)
        assert result64.dtype == torch.float64


class TestSWDtypeMismatch:
    """Test dtype mismatch raises RuntimeError (sw_varients.py:51)."""

    def test_dtype_mismatch_raises_runtime_error(self):
        """Passing float32 and float64 tensors raises RuntimeError (line 51)."""
        x = torch.randn(2, 10, 3, dtype=torch.float32)
        y = torch.randn(2, 10, 3, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="Different dtypes"):
            sw_varients.compute_practical_moments_sw(x, y)


class TestASWDRemoveHistory:
    """Tests for AdaptiveSlicedWassersteinDistance.remove_history (lines 280, 282->exit)."""

    def test_remove_history_none_is_noop(self):
        """remove_history with projs_history=None returns immediately (line 280)."""
        aswd = sw_varients.AdaptiveSlicedWassersteinDistance(projs_history=None)
        aswd.remove_history()  # no-op, early return at line 280

    def test_remove_history_nonexistent_file_is_noop(self, tmp_path):
        """remove_history with a path that doesn't exist returns without error (282->exit)."""
        path = str(tmp_path / "never_created.txt")
        aswd = sw_varients.AdaptiveSlicedWassersteinDistance(projs_history=path)
        # file.exists() is False → branch 282->exit is taken, no os.remove called
        aswd.remove_history()

    def test_remove_history_file_not_found_on_remove(self, tmp_path):
        """FileNotFoundError from os.remove is silently swallowed (lines 286-288)."""
        from unittest.mock import patch
        path = str(tmp_path / "exists.txt")
        open(path, "w").close()  # create the file so file.exists() is True
        aswd = sw_varients.AdaptiveSlicedWassersteinDistance(projs_history=path)
        # Simulate race condition: file disappears between exists() and os.remove()
        with patch("zreg.distance_metrics.sw_varients.os.remove", side_effect=FileNotFoundError):
            aswd.remove_history()  # must not raise


class TestBaseWDForwardStub:
    """Call BaseWD._forward directly to exercise its body (line 180->exit branch)."""

    def test_base_forward_returns_none(self):
        """Directly invoking BaseWD._forward() returns None (the ... stub body)."""
        import torch
        base = sw_varients.BaseWD(nobatchdim=True, device="cpu")
        result = base._forward(torch.randn(4, 3), torch.randn(4, 3))
        assert result is None


# ---------------------------------------------------------------------------
# Coverage gaps: sw_varients.py lines 57, 175, 250-251, 379
# ---------------------------------------------------------------------------


class TestComputePracticalMomentsDimMismatch:
    """sw_varients.py:57 — ValueError when y.size(2) != x.size(2)."""

    def test_dim_mismatch_raises_value_error(self):
        """compute_practical_moments_sw raises ValueError when y has different dim than x."""
        x = torch.randn(2, 10, 3)
        y = torch.randn(2, 10, 2)  # dim=2 vs dim=3
        with pytest.raises(ValueError, match="dimension"):
            sw_varients.compute_practical_moments_sw(x, y, num_projections=5)


class TestBaseWDForwardNdimCheck:
    """sw_varients.py:175 — ValueError when input has fewer than 2 dimensions."""

    def test_1d_input_raises_value_error(self):
        """BaseWD.forward raises ValueError for 1-D tensors (ndim < 2)."""
        swd = sw_varients.SlicedWassersteinDistance(num_projs=10, device="cpu")
        x = torch.randn(10)  # 1-D
        y = torch.randn(10, 3)
        with pytest.raises(ValueError, match="2 dimensions"):
            swd(x, y)


class TestAdaptiveSWDRankNonZero:
    """sw_varients.py:250-251 — projs_history filename gets rank suffix when OMPI rank != 0."""

    def test_rank_nonzero_appends_rank_to_filename(self, tmp_path):
        """When OMPI_COMM_WORLD_RANK=2, projs_history path gets _rank2 suffix."""
        import os
        history_path = str(tmp_path / "projs.txt")
        env = {**os.environ, "OMPI_COMM_WORLD_RANK": "2"}
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("OMPI_COMM_WORLD_RANK", "2")
            obj = sw_varients.AdaptiveSlicedWassersteinDistance(
                projs_history=history_path,
                device="cpu",
            )
        assert "_rank2" in obj.projs_history


class TestOrthogonalSWDNumProjsConstraint:
    """sw_varients.py:379 — ValueError when num_projs > dim."""

    def test_num_projs_exceeds_dim_raises(self, point_clouds_3d):
        """OrthogonalSlicedWassersteinDistance raises when num_projs > 3 for 3-D data."""
        x, y = point_clouds_3d
        oswd = sw_varients.OrthogonalSlicedWassersteinDistance(num_projs=5, device="cpu")
        with pytest.raises(ValueError, match="num_projs"):
            oswd(x, y)
