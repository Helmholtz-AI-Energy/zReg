"""Tests for zreg.dtw module."""

import math
from unittest.mock import MagicMock

import pytest
import torch
from pathlib import Path
import tempfile

from zreg import dtw
from zreg.dtw import DynamicTimeWarping, DTWResult
from zreg.dataset import zRegPointCloud
from zreg.distances import euclidean_distance
from zreg.pairwise_distance_matrix import create_pairwise_distance_matrix


@pytest.fixture
def small_trajectory_pair():
    """Create very small trajectories for fast testing."""
    x = {}
    for i in range(3):
        x[i] = zRegPointCloud(
            pos=torch.randn(10, 3),
            color=torch.rand(10, 3),
            id=torch.arange(10),
        )

    y = {}
    for i in range(3):
        y[i] = zRegPointCloud(
            pos=torch.randn(10, 3),
            color=torch.rand(10, 3),
            id=torch.arange(10),
        )

    return x, y


@pytest.fixture
def identical_trajectories():
    """Create two identical trajectories for testing perfect alignment."""
    x = {}
    for i in range(4):
        pos = torch.randn(15, 3)
        x[i] = zRegPointCloud(
            pos=pos.clone(),
            color=torch.rand(15, 3),
            id=torch.arange(15),
        )

    # y is identical to x
    y = {}
    for i in range(4):
        y[i] = zRegPointCloud(
            pos=x[i]["pos"].clone(),
            color=x[i]["color"].clone(),
            id=x[i]["id"].clone(),
        )

    return x, y


@pytest.fixture
def sample_cost_matrix():
    """Create a sample cost matrix for testing DTW algorithms directly."""
    # Simple 4x4 cost matrix
    cost = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 2.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    return cost


class TestDTWResult:
    """Tests for DTWResult dataclass."""

    def test_creation(self, sample_cost_matrix):
        """Test DTWResult creation."""
        result = DTWResult(
            cost_matrix=sample_cost_matrix,
            accumulated_cost=sample_cost_matrix,
            warping_path=[(0, 0), (1, 1), (2, 2), (3, 3)],
            distance=4.0,
            rotations=None,
        )

        assert result.cost_matrix is not None
        assert result.distance == 4.0
        assert len(result.warping_path) == 4
        assert result.rotations is None

    def test_creation_with_rotations(self, sample_cost_matrix):
        """Test DTWResult creation with rotations."""
        rotations = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
        result = DTWResult(
            cost_matrix=sample_cost_matrix,
            accumulated_cost=sample_cost_matrix,
            warping_path=[(0, 0), (1, 1), (2, 2), (3, 3)],
            distance=4.0,
            rotations=rotations,
        )

        assert result.rotations is not None
        assert result.rotations.shape == (4, 3, 3)


class TestDynamicTimeWarpingInit:
    """Tests for DynamicTimeWarping initialization."""

    def test_basic_init(self, small_trajectory_pair):
        """Test basic initialization."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(x, y)

        assert dtw_obj.x is x
        assert dtw_obj.y is y
        assert dtw_obj.distance_metric == "swd"
        assert dtw_obj.downsample_method == "random"
        assert dtw_obj.result is None

    def test_init_with_options(self, small_trajectory_pair):
        """Test initialization with custom options."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
            cpd_type="rigid",
            window=2,
            normalize=False,
        )

        assert dtw_obj.distance_metric == "euclidean"
        assert dtw_obj.downsample_method is None
        assert dtw_obj.cpd_type == "rigid"
        assert dtw_obj.window == 2
        assert dtw_obj.normalize is False


class TestAccumulatedCostComputation:
    """Tests for accumulated cost matrix computation."""

    def test_diagonal_cost_matrix(self, sample_cost_matrix):
        """Test accumulated cost on a diagonal-optimal cost matrix."""
        x = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(4)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(4)}

        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")
        accumulated = dtw_obj._compute_accumulated_cost(sample_cost_matrix)

        # Check shape
        assert accumulated.shape == sample_cost_matrix.shape

        # First cell should equal cost matrix
        assert accumulated[0, 0] == sample_cost_matrix[0, 0]

        # Accumulated cost should be >= cost at each cell
        assert (accumulated >= sample_cost_matrix).all()

    def test_accumulated_cost_monotonic(self, sample_cost_matrix):
        """Test that accumulated cost is monotonically increasing along path."""
        x = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(4)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(4)}

        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")
        accumulated = dtw_obj._compute_accumulated_cost(sample_cost_matrix)

        # Along the diagonal, values should be non-decreasing
        diagonal_values = [accumulated[i, i].item() for i in range(4)]
        for i in range(1, len(diagonal_values)):
            assert diagonal_values[i] >= diagonal_values[i - 1]


class TestBacktrace:
    """Tests for warping path backtracing."""

    def test_backtrace_diagonal(self, sample_cost_matrix):
        """Test backtracing on diagonal-optimal matrix."""
        x = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(4)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(4)}

        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")
        accumulated = dtw_obj._compute_accumulated_cost(sample_cost_matrix)
        path = dtw_obj._backtrace(accumulated)

        # Path should start at (0, 0)
        assert path[0] == (0, 0)

        # Path should end at (n-1, m-1)
        n, m = sample_cost_matrix.shape
        assert path[-1] == (n - 1, m - 1)

        # For this diagonal cost matrix, optimal path should be diagonal
        expected_path = [(i, i) for i in range(4)]
        assert path == expected_path

    def test_backtrace_valid_steps(self, sample_cost_matrix):
        """Test that backtrace only uses valid step sizes."""
        x = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(4)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(4)}

        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")
        accumulated = dtw_obj._compute_accumulated_cost(sample_cost_matrix)
        path = dtw_obj._backtrace(accumulated)

        # Check that each step is valid (at most +1 in each direction)
        for i in range(1, len(path)):
            di = path[i][0] - path[i - 1][0]
            dj = path[i][1] - path[i - 1][1]
            assert 0 <= di <= 1
            assert 0 <= dj <= 1
            assert di + dj >= 1  # Must make progress

    def test_backtrace_boundary_row_dead_end_raises(self):
        """Test 3: _backtrace raises ValueError when boundary row cell is inf."""
        # Accumulated cost where the path is forced along row 0, but hits an inf.
        # Build a 3x3 accumulated cost: start at (2,2), j==0 boundary forces i--,
        # but to hit a row-0 dead-end we need i==0 while j>0 with acc[0, j-1]==inf.
        # Shape: 2 rows x 3 cols. When backtracing from (1,2):
        #   i==1>0, j==2>0 -> interior, min of [acc[0,2], acc[1,1], acc[0,1]] = ...
        # Simpler: construct a 1x3 matrix (i==0 throughout) so that j boundary branch
        # is never taken. Instead: 2x2, start (1,1), interior min picks diagonal -> (0,0).
        # To force boundary row dead-end: 2x3, start (1,2).
        #   Interior: candidates = acc[0,2], acc[1,1], acc[0,1]
        #   We want to end up in i==0 state with acc[0, j-1]==inf.
        # Easiest: 1x3 accumulated cost (only row 0 exists). i==0 always, force j boundary.
        # acc[0, j-1] is inf when j==2 -> acc[0,1] is inf.
        x_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        y_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        dtw_obj = DynamicTimeWarping(x_dummy, y_dummy, distance_metric="euclidean")

        # 1x3: start (0,2), i==0 so j-branch: check acc[0,1] = inf -> should raise
        acc = torch.tensor([[1.0, float("inf"), 3.0]])
        with pytest.raises(ValueError, match="DTW warping path could not be traced"):
            dtw_obj._backtrace(acc)

    def test_backtrace_boundary_col_dead_end_raises(self):
        """Test 3b: _backtrace raises ValueError when boundary col cell is inf."""
        x_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        y_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        dtw_obj = DynamicTimeWarping(x_dummy, y_dummy, distance_metric="euclidean")

        # 3x1: start (2,0), j==0 so i-branch: check acc[1,0] = inf -> should raise
        acc = torch.tensor([[1.0], [float("inf")], [3.0]])
        with pytest.raises(ValueError, match="DTW warping path could not be traced"):
            dtw_obj._backtrace(acc)

    def test_backtrace_interior_all_inf_raises(self):
        """Test 4: _backtrace raises ValueError when all interior predecessors are inf."""
        x_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        y_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        dtw_obj = DynamicTimeWarping(x_dummy, y_dummy, distance_metric="euclidean")

        # 3x3 accumulated cost where (2,2) is reachable but all predecessors are inf
        acc = torch.tensor([
            [1.0, float("inf"), float("inf")],
            [float("inf"), float("inf"), float("inf")],
            [float("inf"), float("inf"), 5.0],
        ])
        with pytest.raises(ValueError, match="DTW warping path could not be traced"):
            dtw_obj._backtrace(acc)

    def test_backtrace_interior_error_contains_position(self):
        """Test 4b: ValueError for interior dead-end includes current (i, j) position."""
        x_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        y_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        dtw_obj = DynamicTimeWarping(x_dummy, y_dummy, distance_metric="euclidean")

        acc = torch.tensor([
            [1.0, float("inf"), float("inf")],
            [float("inf"), float("inf"), float("inf")],
            [float("inf"), float("inf"), 5.0],
        ])
        with pytest.raises(ValueError, match=r"\(2, 2\)"):
            dtw_obj._backtrace(acc)

    def test_backtrace_1x1_returns_origin(self):
        """Test 5: _backtrace on a 1x1 matrix returns [(0, 0)] with no error."""
        x_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        y_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        dtw_obj = DynamicTimeWarping(x_dummy, y_dummy, distance_metric="euclidean")

        acc = torch.tensor([[2.5]])
        path = dtw_obj._backtrace(acc)
        assert path == [(0, 0)]

    def test_backtrace_windowed_valid_path(self):
        """Test 2: _backtrace on a windowed accumulated cost returns correct path."""
        x_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        y_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        dtw_obj = DynamicTimeWarping(x_dummy, y_dummy, distance_metric="euclidean")

        # 3x3 windowed cost: cells outside window are inf, diagonal path is valid
        acc = torch.tensor([
            [1.0, 2.0, float("inf")],
            [3.0, 3.0, 5.0],
            [float("inf"), 5.0, 6.0],
        ])
        path = dtw_obj._backtrace(acc)
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

    def test_backtrace_argmin1_j_decrement(self):
        """Interior step with argmin==1 decrements j only (line 298: j -= 1)."""
        x_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        y_dummy = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        dtw_obj = DynamicTimeWarping(x_dummy, y_dummy, distance_metric="euclidean")

        # At (2,2): acc[1,2]=4.0, acc[2,1]=1.5, acc[1,1]=3.0 → argmin=1 → j -= 1
        acc = torch.tensor([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 1.5, 5.0],
        ])
        path = dtw_obj._backtrace(acc)
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        # (2,2) → (2,1) via j-decrement, so (2,1) must be in the path
        assert (2, 1) in path


class TestDTWCompute:
    """Tests for full DTW computation."""

    def test_compute_euclidean(self, small_trajectory_pair):
        """Test full DTW computation with Euclidean distance."""
        x, y = small_trajectory_pair

        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
            normalize=True,
        )
        result = dtw_obj.compute()

        assert isinstance(result, DTWResult)
        assert result.cost_matrix is not None
        assert result.accumulated_cost is not None
        assert len(result.warping_path) > 0
        assert result.distance >= 0

    def test_compute_stores_result(self, small_trajectory_pair):
        """Test that compute stores result in instance."""
        x, y = small_trajectory_pair

        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )

        assert dtw_obj.result is None
        result = dtw_obj.compute()
        assert dtw_obj.result is result

    def test_compute_with_swd(self, small_trajectory_pair):
        """Test DTW computation with SWD distance."""
        x, y = small_trajectory_pair

        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="swd",
            downsample_method="random",
            normalize=True,
        )
        result = dtw_obj.compute()

        assert result.distance >= 0
        assert len(result.warping_path) >= 3  # At least length of shorter trajectory

    def test_compute_with_window(self, small_trajectory_pair):
        """Test DTW computation with windowing."""
        x, y = small_trajectory_pair

        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
            window=1,
        )
        result = dtw_obj.compute()

        assert result.distance >= 0
        # With window=1, path should stay close to diagonal
        for i, j in result.warping_path:
            assert abs(i - j) <= 1


class TestDTWAccessors:
    """Tests for DTW accessor methods."""

    def test_get_warping_path_before_compute(self, small_trajectory_pair):
        """Test that get_warping_path raises before compute."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")

        with pytest.raises(RuntimeError):
            dtw_obj.get_warping_path()

    def test_get_warping_path_after_compute(self, small_trajectory_pair):
        """Test get_warping_path after compute."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        dtw_obj.compute()

        path = dtw_obj.get_warping_path()
        assert len(path) > 0
        assert path[0] == (0, 0)

    def test_get_distance_before_compute(self, small_trajectory_pair):
        """Test that get_distance raises before compute."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")

        with pytest.raises(RuntimeError):
            dtw_obj.get_distance()

    def test_get_distance_after_compute(self, small_trajectory_pair):
        """Test get_distance after compute."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        result = dtw_obj.compute()

        distance = dtw_obj.get_distance()
        assert distance == result.distance

    def test_get_aligned_indices(self, small_trajectory_pair):
        """Test get_aligned_indices."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        dtw_obj.compute()

        x_indices, y_indices = dtw_obj.get_aligned_indices()
        assert len(x_indices) == len(y_indices)
        assert len(x_indices) == len(dtw_obj.get_warping_path())


class TestGetAlignedTrajectory:
    """Tests for trajectory alignment."""

    def test_aligned_trajectory_reference_x(self, small_trajectory_pair):
        """Test aligning trajectory with x as reference."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        dtw_obj.compute()

        aligned = dtw_obj.get_aligned_trajectory(y, reference="x")

        # Should have same time indices as x
        assert set(aligned.keys()).issubset(set(x.keys()))

    def test_aligned_trajectory_reference_y(self, small_trajectory_pair):
        """Test aligning trajectory with y as reference."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        dtw_obj.compute()

        aligned = dtw_obj.get_aligned_trajectory(x, reference="y")

        # Should have same time indices as y
        assert set(aligned.keys()).issubset(set(y.keys()))

    def test_aligned_trajectory_invalid_reference(self, small_trajectory_pair):
        """Test that invalid reference raises error."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        dtw_obj.compute()

        with pytest.raises(ValueError):
            dtw_obj.get_aligned_trajectory(y, reference="z")

    def test_aligned_trajectory_before_compute(self, small_trajectory_pair):
        """Test that get_aligned_trajectory raises before compute."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")

        with pytest.raises(RuntimeError):
            dtw_obj.get_aligned_trajectory(y)


class TestSetCostMatrix:
    """Tests for setting precomputed cost matrix."""

    def test_set_cost_matrix(self, small_trajectory_pair, sample_cost_matrix):
        """Test setting a precomputed cost matrix."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")

        # Resize cost matrix to match trajectory lengths
        cost = torch.randn(3, 3).abs()  # 3x3 for 3 time points each
        dtw_obj.set_cost_matrix(cost)

        assert dtw_obj._cost_matrix is not None
        assert torch.equal(dtw_obj._cost_matrix, cost)

    def test_compute_with_precomputed_matrix(self, small_trajectory_pair):
        """Test that compute uses precomputed cost matrix."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )

        # Set a custom cost matrix
        cost = torch.ones(3, 3)
        dtw_obj.set_cost_matrix(cost)

        result = dtw_obj.compute()

        # Should use the precomputed matrix
        assert torch.equal(result.cost_matrix, cost)


class TestSaveLoad:
    """Tests for saving and loading DTW results."""

    def test_save_and_load(self, small_trajectory_pair):
        """Test saving and loading DTW results."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        result = dtw_obj.compute()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dtw_result.pt"
            dtw_obj.save(path)

            assert path.exists()

            loaded_result = DynamicTimeWarping.load(path)

            assert isinstance(loaded_result, DTWResult)
            assert torch.equal(loaded_result.cost_matrix, result.cost_matrix)
            assert loaded_result.distance == result.distance
            assert loaded_result.warping_path == result.warping_path

    def test_save_before_compute(self, small_trajectory_pair):
        """Test that save raises before compute."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dtw_result.pt"
            with pytest.raises(RuntimeError):
                dtw_obj.save(path)


class TestPlotAlignment:
    """Tests for plotting functionality."""

    def test_plot_before_compute(self, small_trajectory_pair):
        """Test that plot raises before compute."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")

        with pytest.raises(RuntimeError):
            dtw_obj.plot_alignment()

    def test_plot_saves_file(self, small_trajectory_pair):
        """Test that plot saves to file when path given."""
        pytest.importorskip("matplotlib")

        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        dtw_obj.compute()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.png"
            dtw_obj.plot_alignment(save_path=path)

            assert path.exists()


class TestExports:
    """Tests for module exports."""

    def test_exports(self):
        """Test that expected classes are exported."""
        assert "DynamicTimeWarping" in dtw.__all__
        assert "DTWResult" in dtw.__all__

    def test_import_from_zreg(self):
        """Test that dtw module is accessible from zreg."""
        import zreg
        assert hasattr(zreg, "dtw")
        assert hasattr(zreg.dtw, "DynamicTimeWarping")
        assert hasattr(zreg.dtw, "DTWResult")

    def test_import_dtwresult_from_package(self):
        """Test that DTWResult is importable from dtw package."""
        from zreg.dtw import DTWResult
        from zreg.dtw.result import DTWResult as DTWResultDirect

        assert DTWResult is DTWResultDirect

    def test_import_compose_constraints(self):
        """Test that compose_constraints is importable from dtw package."""
        from zreg.dtw import compose_constraints
        from zreg.dtw.constraints import compose_constraints as direct

        assert compose_constraints is direct


class TestDTWMetricsAndBoundaries:
    """Tests for DTW with additional distance metrics, windowed asymmetric inputs, and boundary conditions (TEST-02)."""

    def test_compute_manhattan_metric(self, small_trajectory_pair):
        """Test DTW computation with manhattan distance metric."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="manhattan",
            downsample_method=None,
            normalize=True,
        )
        result = dtw_obj.compute()

        assert isinstance(result, DTWResult)
        assert result.distance >= 0
        assert len(result.warping_path) >= 3
        assert result.warping_path[0] == (0, 0)
        assert result.warping_path[-1] == (2, 2)

    def test_compute_cpd_metric(self, small_trajectory_pair):
        """Test DTW computation with cpd distance metric (requires cpd_type for registration)."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="cpd",
            downsample_method=None,
            cpd_type="rigid",
        )
        result = dtw_obj.compute()

        assert isinstance(result, DTWResult)
        # CPD quality metric (reg.q) can be negative, so check finite instead of >= 0
        assert math.isfinite(result.distance)
        assert len(result.warping_path) >= 3
        assert result.warping_path[0] == (0, 0)
        assert result.warping_path[-1] == (2, 2)

    def test_compute_minkowski_metric(self, small_trajectory_pair):
        """Test DTW computation with minkowski distance metric."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="minkowski",
            downsample_method=None,
            normalize=True,
        )
        result = dtw_obj.compute()

        assert isinstance(result, DTWResult)
        assert result.distance >= 0
        assert len(result.warping_path) >= 3
        assert result.warping_path[0] == (0, 0)
        assert result.warping_path[-1] == (2, 2)

    def test_windowed_asymmetric_trajectories(self):
        """Test DTW with windowed constraint on asymmetric trajectory lengths."""
        # x has 4 time points, y has 3 time points
        x = {i: zRegPointCloud(pos=torch.randn(10, 3)) for i in range(4)}
        y = {i: zRegPointCloud(pos=torch.randn(10, 3)) for i in range(3)}

        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
            window=1,
        )
        result = dtw_obj.compute()

        assert result.warping_path[0] == (0, 0)
        assert result.warping_path[-1] == (3, 2)
        assert result.distance >= 0

        # Each step must respect window constraint: |i - j| <= 1
        for i, j in result.warping_path:
            assert abs(i - j) <= 1, f"Window constraint violated at ({i}, {j})"

        # Each consecutive step must be a valid DTW move
        for k in range(1, len(result.warping_path)):
            di = result.warping_path[k][0] - result.warping_path[k - 1][0]
            dj = result.warping_path[k][1] - result.warping_path[k - 1][1]
            assert 0 <= di <= 1
            assert 0 <= dj <= 1
            assert di + dj >= 1

    def test_boundary_single_timepoint_x(self):
        """Test DTW with single time point in x (1xN boundary)."""
        x = {0: zRegPointCloud(pos=torch.randn(10, 3))}
        y = {i: zRegPointCloud(pos=torch.randn(10, 3)) for i in range(3)}

        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        result = dtw_obj.compute()

        assert result.warping_path[0] == (0, 0)
        assert result.warping_path[-1] == (0, 2)

        # All steps must have i == 0 (forced along row boundary)
        for i, j in result.warping_path:
            assert i == 0, f"Expected i=0 for single-timepoint x, got ({i}, {j})"

    def test_boundary_single_timepoint_y(self):
        """Test DTW with single time point in y (Nx1 boundary)."""
        x = {i: zRegPointCloud(pos=torch.randn(10, 3)) for i in range(3)}
        y = {0: zRegPointCloud(pos=torch.randn(10, 3))}

        dtw_obj = DynamicTimeWarping(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
        )
        result = dtw_obj.compute()

        assert result.warping_path[0] == (0, 0)
        assert result.warping_path[-1] == (2, 0)

        # All steps must have j == 0 (forced along column boundary)
        for i, j in result.warping_path:
            assert j == 0, f"Expected j=0 for single-timepoint y, got ({i}, {j})"

    def test_boundary_equal_identical_trajectories(self):
        """Test DTW with identical trajectories produces diagonal path.

        Uses a precomputed cost matrix where diagonal is zero to verify
        that DTW correctly identifies the optimal diagonal alignment
        when self-match cost is zero.
        """
        # Build trajectories (needed for DynamicTimeWarping constructor)
        x = {i: zRegPointCloud(pos=torch.randn(10, 3)) for i in range(4)}
        y = {i: zRegPointCloud(pos=torch.randn(10, 3)) for i in range(4)}

        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")

        # Set a cost matrix where diagonal is 0 (perfect self-match)
        # and off-diagonal > 0 (non-zero cross-match cost)
        cost = torch.tensor([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ])
        dtw_obj.set_cost_matrix(cost)
        result = dtw_obj.compute()

        # Path should be the diagonal
        expected_path = [(i, i) for i in range(4)]
        assert result.warping_path == expected_path, (
            f"Expected diagonal path {expected_path}, got {result.warping_path}"
        )

        # Distance should be 0 for perfect diagonal alignment
        assert result.distance == 0.0


class TestComposeConstraints:
    """Tests for compose_constraints utility function."""

    def test_no_constraints_allows_all(self):
        """Test that no constraints means all cells allowed."""
        from zreg.dtw.constraints import compose_constraints

        composed = compose_constraints()
        # All cells should be allowed
        assert composed(0, 0, 10, 10) is True
        assert composed(5, 5, 10, 10) is True
        assert composed(9, 9, 10, 10) is True

    def test_single_constraint_passthrough(self):
        """Test that single constraint is passed through correctly."""
        from zreg.dtw.constraints import compose_constraints

        def sakoe_chiba(i, j, n, m, window=2):
            return abs(i - j) <= window

        composed = compose_constraints(lambda i, j, n, m: sakoe_chiba(i, j, n, m, window=1))

        # Within window
        assert composed(0, 0, 10, 10) is True
        assert composed(1, 0, 10, 10) is True
        assert composed(0, 1, 10, 10) is True

        # Outside window
        assert composed(0, 2, 10, 10) is False
        assert composed(5, 8, 10, 10) is False

    def test_multiple_constraints_intersection(self):
        """Test that multiple constraints are ANDed together."""
        from zreg.dtw.constraints import compose_constraints

        # Constraint 1: i >= 2
        constraint1 = lambda i, j, n, m: i >= 2
        # Constraint 2: j >= 3
        constraint2 = lambda i, j, n, m: j >= 3

        composed = compose_constraints(constraint1, constraint2)

        # Both satisfied
        assert composed(2, 3, 10, 10) is True
        assert composed(5, 5, 10, 10) is True

        # Only first satisfied
        assert composed(2, 2, 10, 10) is False

        # Only second satisfied
        assert composed(1, 3, 10, 10) is False

        # Neither satisfied
        assert composed(1, 2, 10, 10) is False

    def test_constraint_receives_matrix_dimensions(self):
        """Test that constraints receive correct n and m values."""
        from zreg.dtw.constraints import compose_constraints

        received_args = []

        def capture_args(i, j, n, m):
            received_args.append((i, j, n, m))
            return True

        composed = compose_constraints(capture_args)
        composed(3, 4, 10, 15)

        assert received_args == [(3, 4, 10, 15)]


class TestAlignTrajectoryDuplicateIndex:
    """Tests for get_aligned_trajectory with duplicate warping-path indices (408->407, 413->412)."""

    def _make_dtw_with_path(self, path):
        """Return a DynamicTimeWarping object with a pre-set result using the given path."""
        n_x = max(p[0] for p in path) + 1
        n_y = max(p[1] for p in path) + 1
        x = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(n_x)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3)) for i in range(n_y)}
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean")

        cost = torch.ones(n_x, n_y)
        acc = torch.cumsum(torch.cumsum(cost, dim=0), dim=1)
        dtw_obj.result = DTWResult(
            cost_matrix=cost,
            accumulated_cost=acc,
            warping_path=path,
            distance=float(acc[-1, -1]),
            rotations=None,
        )
        return dtw_obj, x, y

    def test_reference_x_duplicate_x_idx_skipped(self):
        """Second occurrence of same x_idx is skipped (line 408 False → 408->407 covered)."""
        # Path: (0,0), (0,1), (1,2) — x_idx=0 appears twice
        path = [(0, 0), (0, 1), (1, 2)]
        dtw_obj, x, y = self._make_dtw_with_path(path)

        aligned = dtw_obj.get_aligned_trajectory(y, reference="x")
        # Only 2 unique x_indices: 0 and 1
        assert set(aligned.keys()) == {0, 1}
        # x_idx=0 gets y[0] (first match), not y[1]
        assert aligned[0] is y[0]

    def test_reference_y_duplicate_y_idx_skipped(self):
        """Second occurrence of same y_idx is skipped (line 413 False → 413->412 covered)."""
        # Path: (0,0), (1,0), (2,1) — y_idx=0 appears twice
        path = [(0, 0), (1, 0), (2, 1)]
        dtw_obj, x, y = self._make_dtw_with_path(path)

        aligned = dtw_obj.get_aligned_trajectory(x, reference="y")
        # Only 2 unique y_indices: 0 and 1
        assert set(aligned.keys()) == {0, 1}
        # y_idx=0 gets x[0] (first match), not x[1]
        assert aligned[0] is x[0]


class TestPlotAlignmentNoMatplotlib:
    """Tests for plot_alignment when matplotlib is unavailable (lines 447-451)."""

    def test_plot_alignment_returns_early_without_matplotlib(self, caplog):
        """plot_alignment returns early and logs a warning when matplotlib is missing (lines 447-451)."""
        import logging
        x = {i: zRegPointCloud(pos=torch.randn(5, 3), color=torch.rand(5, 3), id=torch.arange(5)) for i in range(3)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3), color=torch.rand(5, 3), id=torch.arange(5)) for i in range(3)}
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean", downsample_method=None)
        dtw_obj.compute()

        try:
            import matplotlib  # noqa: F401
            pytest.skip("matplotlib IS available — early-return path not taken")
        except ImportError:
            pass

        with caplog.at_level(logging.WARNING, logger="zreg.dtw.core"):
            dtw_obj.plot_alignment()  # should return without raising

        assert any("matplotlib" in str(r.message) for r in caplog.records)

    def test_plot_alignment_with_mocked_matplotlib(self, tmp_path):
        """plot_alignment runs the full plotting code path when matplotlib is available (lines 453-495)."""
        from unittest.mock import MagicMock, patch

        x = {i: zRegPointCloud(pos=torch.randn(5, 3), color=torch.rand(5, 3), id=torch.arange(5)) for i in range(3)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3), color=torch.rand(5, 3), id=torch.arange(5)) for i in range(3)}
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean", downsample_method=None)
        dtw_obj.compute()

        original_result = dtw_obj.result

        # Build a 2D-cost variant so the `if cost.ndim == 3:` False branch (454->457) is also taken.
        result_2d = DTWResult(
            cost_matrix=original_result.cost_matrix[0],  # 2D slice
            accumulated_cost=original_result.accumulated_cost,
            warping_path=original_result.warping_path,
            distance=original_result.distance,
            rotations=original_result.rotations,
        )

        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_ax1, mock_ax2 = MagicMock(), MagicMock()
        mock_plt.subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        # Wire mock_mpl.pyplot = mock_plt so `import matplotlib.pyplot as plt` gets mock_plt.
        mock_mpl = MagicMock()
        mock_mpl.pyplot = mock_plt

        save_path = str(tmp_path / "dtw_plot.png")
        with patch.dict("sys.modules", {"matplotlib": mock_mpl, "matplotlib.pyplot": mock_plt}):
            # 3D cost matrix → line 455 (cost = cost[metric_index]) is executed
            dtw_obj.result = original_result
            dtw_obj.plot_alignment(save_path=save_path)
            # 2D cost matrix → False branch (454->457) is taken; also covers plt.show (line 493)
            dtw_obj.result = result_2d
            dtw_obj.plot_alignment()

        assert mock_plt.subplots.call_count == 2
        mock_plt.savefig.assert_called_once()


class TestPlotAlignmentMatplotlibImportError:
    """Force the matplotlib ImportError path (lines 449-451) via sys.modules patching."""

    def test_plot_returns_early_when_matplotlib_pyplot_absent(self, caplog):
        """Setting matplotlib.pyplot=None in sys.modules triggers ImportError (lines 449-451)."""
        import logging
        from unittest.mock import patch

        x = {i: zRegPointCloud(pos=torch.randn(5, 3), color=torch.rand(5, 3), id=torch.arange(5)) for i in range(3)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3), color=torch.rand(5, 3), id=torch.arange(5)) for i in range(3)}
        dtw_obj = DynamicTimeWarping(x, y, distance_metric="euclidean", downsample_method=None)
        dtw_obj.compute()

        with caplog.at_level(logging.WARNING, logger="zreg.dtw.core"):
            with patch.dict("sys.modules", {"matplotlib.pyplot": None}):
                dtw_obj.plot_alignment()


class TestAnnotationWideningRED:
    """RED-phase tests: DynamicTimeWarping accepts DistanceMetric callables.

    These tests define the type-widening behaviour required by DTW-02. They
    intentionally fail before the annotation is updated in dtw/core.py.
    """

    def test_dtw_annotation_includes_distance_metric_protocol(self):
        """DynamicTimeWarping.__init__ annotation for distance_metric must include DistanceMetric."""
        hints = DynamicTimeWarping.__init__.__annotations__
        annotation_str = str(hints.get("distance_metric", ""))
        assert "DistanceMetric" in annotation_str, (
            f"Expected 'DistanceMetric' in annotation, got: {annotation_str!r}"
        )

    def test_dtw_accepts_callable_without_type_error(self):
        """DynamicTimeWarping must accept a callable distance_metric without raising."""
        from zreg.distances import euclidean_distance
        x = {i: zRegPointCloud(pos=torch.randn(5, 3), id=torch.arange(5)) for i in range(2)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3), id=torch.arange(5)) for i in range(2)}
        d = DynamicTimeWarping(x, y, distance_metric=euclidean_distance, downsample_method=None)
        assert callable(d.distance_metric)


class TestCallableMetric:
    """End-to-end tests covering callable distance_metric pass-through (DTW-02)."""

    def test_callable_metric_in_create_pairwise_distance_matrix(self, small_trajectory_pair):
        """create_pairwise_distance_matrix accepts a callable metric and returns a float tensor."""
        x, y = small_trajectory_pair
        matrix = create_pairwise_distance_matrix(
            x, y, distance_metric=euclidean_distance, downsample_method=None
        ).cost_matrix
        assert isinstance(matrix, torch.Tensor)
        assert matrix.is_floating_point()
        # Output shape is (num_metrics, len_x, len_y); with a single metric this is (1, N, M)
        assert matrix.shape[-2] == len(x)
        assert matrix.shape[-1] == len(y)

    def test_callable_metric_matches_string_metric(self, small_trajectory_pair):
        """DynamicTimeWarping with callable metric yields same result as equivalent string."""
        x, y = small_trajectory_pair
        dtw_callable = DynamicTimeWarping(
            x, y, distance_metric=euclidean_distance, downsample_method=None
        )
        dtw_string = DynamicTimeWarping(
            x, y, distance_metric="euclidean", downsample_method=None
        )
        result_callable = dtw_callable.compute()
        result_string = dtw_string.compute()
        assert result_callable.distance >= 0
        assert result_string.distance >= 0
        assert result_callable.distance == pytest.approx(result_string.distance, rel=1e-5)

    def test_callable_is_actually_invoked(self, small_trajectory_pair):
        """The callable metric must actually be called during distance computation."""
        x, y = small_trajectory_pair
        mock = MagicMock(wraps=euclidean_distance)
        create_pairwise_distance_matrix(
            x, y, distance_metric=mock, downsample_method=None
        )
        assert mock.call_count >= 1

    def test_custom_lambda_metric_returns_zeros(self, small_trajectory_pair):
        """A lambda returning zeros produces an all-zero distance matrix."""
        x, y = small_trajectory_pair
        zero_metric = lambda x, y, **kw: torch.zeros(1)  # noqa: E731
        result = create_pairwise_distance_matrix(
            x, y, distance_metric=zero_metric, downsample_method=None
        )
        assert torch.all(result.cost_matrix == 0)


class TestDTWResultStoredTransforms:
    """Tests for DTWResult.stored_transforms field (ALIGN-03, D-07)."""

    def test_dtw_result_has_stored_transforms_field(self):
        """DTWResult must have a stored_transforms field with default empty dict."""
        cost = torch.zeros(3, 3)
        acc = torch.zeros(3, 3)
        result = DTWResult(
            cost_matrix=cost,
            accumulated_cost=acc,
            warping_path=[(0, 0), (1, 1), (2, 2)],
            distance=0.0,
        )
        assert hasattr(result, "stored_transforms")
        assert isinstance(result.stored_transforms, dict)
        assert len(result.stored_transforms) == 0

    def test_dtw_result_stored_transforms_independent_instances(self):
        """Two DTWResult instances must have independent stored_transforms dicts (no shared mutable default)."""
        cost = torch.zeros(2, 2)
        acc = torch.zeros(2, 2)
        r1 = DTWResult(cost_matrix=cost, accumulated_cost=acc, warping_path=[(0, 0), (1, 1)], distance=0.0)
        r2 = DTWResult(cost_matrix=cost, accumulated_cost=acc, warping_path=[(0, 0), (1, 1)], distance=0.0)
        r1.stored_transforms[(0, 0)] = "sentinel"
        assert (0, 0) not in r2.stored_transforms, "stored_transforms instances must not be shared"

    def test_dtw_result_accepts_stored_transforms_kwarg(self):
        """DTWResult can be constructed with an explicit stored_transforms dict."""
        from zreg.types import StoredTransform
        cost = torch.zeros(2, 2)
        acc = torch.zeros(2, 2)
        st = StoredTransform(
            transform=object(),
            src_min=torch.tensor(0.0),
            src_max=torch.tensor(1.0),
            tgt_min=torch.tensor(0.0),
            tgt_max=torch.tensor(1.0),
        )
        transforms = {(0, 0): st}
        result = DTWResult(
            cost_matrix=cost,
            accumulated_cost=acc,
            warping_path=[(0, 0), (1, 1)],
            distance=0.0,
            stored_transforms=transforms,
        )
        assert result.stored_transforms is transforms
        assert (0, 0) in result.stored_transforms

    def test_dtw_compute_threads_stored_transforms(self, small_trajectory_pair):
        """DynamicTimeWarping.compute() with cpd_type='rigid' produces non-empty stored_transforms."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y, distance_metric="euclidean", downsample_method="random", cpd_type="rigid"
        )
        result = dtw_obj.compute()
        assert hasattr(result, "stored_transforms")
        assert isinstance(result.stored_transforms, dict)
        assert len(result.stored_transforms) > 0

    def test_dtw_compute_empty_stored_transforms_without_cpd(self, small_trajectory_pair):
        """DynamicTimeWarping.compute() with cpd_type=None produces empty stored_transforms."""
        x, y = small_trajectory_pair
        dtw_obj = DynamicTimeWarping(
            x, y, distance_metric="euclidean", downsample_method=None, cpd_type=None
        )
        result = dtw_obj.compute()
        assert isinstance(result.stored_transforms, dict)
        assert len(result.stored_transforms) == 0
