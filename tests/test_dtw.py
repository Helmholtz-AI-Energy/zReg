"""Tests for zreg.dtw module."""

import pytest
import torch
from pathlib import Path
import tempfile

from zreg import dtw
from zreg.dtw import DynamicTimeWarping, DTWResult
from zreg.dataset import zRegPointCloud


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
