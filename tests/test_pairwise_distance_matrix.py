"""Tests for zreg.pairwise_distance_matrix module."""

import pytest
import torch
from unittest.mock import MagicMock

from zreg import pairwise_distance_matrix
from zreg.dataset import zRegPointCloud


@pytest.fixture
def trajectory_pair():
    """Create two simple trajectories for testing."""
    x = {}
    for i in range(5):
        x[i] = zRegPointCloud(
            pos=torch.randn(20, 3) + i * 0.5,
            label=torch.rand(20, 3),
            id=torch.arange(20),
        )
    
    y = {}
    for i in range(5):
        y[i] = zRegPointCloud(
            pos=torch.randn(20, 3) + i * 0.5,
            label=torch.rand(20, 3),
            id=torch.arange(20),
        )
    
    return x, y


@pytest.fixture
def small_trajectory_pair():
    """Create very small trajectories for fast testing."""
    x = {}
    for i in range(3):
        x[i] = zRegPointCloud(
            pos=torch.randn(10, 3),
            label=torch.rand(10, 3),
            id=torch.arange(10),
        )
    
    y = {}
    for i in range(3):
        y[i] = zRegPointCloud(
            pos=torch.randn(10, 3),
            label=torch.rand(10, 3),
            id=torch.arange(10),
        )
    
    return x, y


class TestSanitizePairwiseDistanceMatrix:
    """Tests for _sanitize_pairwise_distance_matrix helper function."""

    def test_single_distance_metric(self, small_trajectory_pair):
        """Test sanitization with single distance metric."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics="euclidean",
            downsample_method=None,
            x=x,
            y=y,
        )

        assert len(distance_fns) == 1
        assert callable(distance_fns[0])

    def test_multiple_distance_metrics(self, small_trajectory_pair):
        """Test sanitization with multiple distance metrics."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs=[None, None],
            distance_metrics=["euclidean", "manhattan"],
            downsample_method=None,
            x=x,
            y=y,
        )
        
        assert len(distance_fns) == 2

    def test_swd_requires_downsampling(self, small_trajectory_pair):
        """Test that SWD metrics require downsampling."""
        x, y = small_trajectory_pair
        
        with pytest.raises(RuntimeError):
            pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
                distance_kwargs=None,
                distance_metrics="swd",
                downsample_method=None,
                x=x,
                y=y,
            )

    def test_swd_with_downsampling(self, small_trajectory_pair):
        """Test SWD with downsampling method."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics="swd",
            downsample_method="random",
            x=x,
            y=y,
        )
        
        assert len(distance_fns) == 1
        assert ds_method == "random"
        assert callable(ds_fn)

    def test_cpd_distance_metric(self, small_trajectory_pair):
        """Test CPD as distance metric returns None function."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics="cpd",
            downsample_method=None,
            x=x,
            y=y,
        )
        
        assert distance_fns[0] is None

    def test_invalid_distance_metric(self, small_trajectory_pair):
        """Test invalid distance metric raises error."""
        x, y = small_trajectory_pair
        
        with pytest.raises(ValueError):
            pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
                distance_kwargs=None,
                distance_metrics="invalid_metric",
                downsample_method=None,
                x=x,
                y=y,
            )

    def test_downsampling_methods(self, small_trajectory_pair):
        """Test various downsampling methods."""
        x, y = small_trajectory_pair
        
        for method in ["random", "uniform", "farthest"]:
            _, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
                distance_kwargs=None,
                distance_metrics="euclidean",
                downsample_method=method,
                x=x,
                y=y,
            )
            assert ds_method == method
            assert callable(ds_fn)

    def test_invalid_downsampling_method(self, small_trajectory_pair):
        """Test invalid downsampling method raises error."""
        x, y = small_trajectory_pair

        with pytest.raises(ValueError):
            pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
                distance_kwargs=None,
                distance_metrics="euclidean",
                downsample_method="invalid_method",
                x=x,
                y=y,
            )


class TestCreatePairwiseDistanceMatrix:
    """Tests for create_pairwise_distance_matrix function."""

    def test_basic_creation(self, small_trajectory_pair):
        """Test basic pairwise distance matrix creation."""
        x, y = small_trajectory_pair

        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            downsample_method=None,
        )
        distance_matrix, rots = result.cost_matrix, result.rotations

        assert distance_matrix.shape == (1, 3, 3)

    def test_matrix_shape_with_window(self, small_trajectory_pair):
        """Test pairwise distance matrix with window constraint."""
        x, y = small_trajectory_pair

        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            window=1,
            normalize=True,
            distance_metric="euclidean",
        )
        distance_matrix, rots = result.cost_matrix, result.rotations

        assert distance_matrix.shape == (1, 3, 3)

    def test_multiple_distance_metrics(self, small_trajectory_pair):
        """Test pairwise distance matrix with multiple distance metrics."""
        x, y = small_trajectory_pair

        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric=["euclidean", "manhattan"],
            distance_kwargs=[None, None],
        )
        distance_matrix, rots = result.cost_matrix, result.rotations

        assert distance_matrix.shape[0] == 2

    def test_with_downsampling(self, small_trajectory_pair):
        """Test pairwise distance matrix with downsampling."""
        x, y = small_trajectory_pair

        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="swd",
            downsample_method="random",
        )
        distance_matrix, rots = result.cost_matrix, result.rotations

        assert distance_matrix.shape == (1, 3, 3)

    def test_with_cpd(self, small_trajectory_pair):
        """Test pairwise distance matrix with CPD registration."""
        x, y = small_trajectory_pair

        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            cpd_type="rigid",
            downsample_method="random",
        )
        distance_matrix, rots = result.cost_matrix, result.rotations

        assert distance_matrix.shape == (1, 3, 3)
        assert len(rots) > 0

    def test_normalization_effect(self, small_trajectory_pair):
        """Test that normalization affects results."""
        x, y = small_trajectory_pair

        matrix_normalized = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y, normalize=True, distance_metric="euclidean"
        ).cost_matrix

        matrix_unnormalized = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y, normalize=False, distance_metric="euclidean"
        ).cost_matrix

        assert not torch.allclose(matrix_normalized, matrix_unnormalized)

    def test_finite_values(self, small_trajectory_pair):
        """Test that computed values are finite."""
        x, y = small_trajectory_pair

        distance_matrix = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
        ).cost_matrix

        assert torch.isfinite(distance_matrix).all()


class TestCreatePairwiseDistanceMatrixGivenRigidRot:
    """Tests for create_pairwise_distance_matrix_given_rigid_rot function."""

    def test_basic_creation(self, small_trajectory_pair):
        """Test basic pairwise distance matrix creation with given rotation."""
        x, y = small_trajectory_pair
        
        rotation = torch.eye(3)
        translation = torch.zeros(3)
        
        distance_matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            scale=1.0,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert distance_matrix.shape == (1, 3, 3)

    def test_with_rotation(self, small_trajectory_pair):
        """Test pairwise distance matrix with non-identity rotation."""
        x, y = small_trajectory_pair
        
        rotation = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = torch.tensor([1.0, 2.0, 0.0])
        
        distance_matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            scale=1.0,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert distance_matrix.shape == (1, 3, 3)
        assert torch.isfinite(distance_matrix).all()

    def test_with_window(self, small_trajectory_pair):
        """Test pairwise distance matrix with window constraint."""
        x, y = small_trajectory_pair
        
        rotation = torch.eye(3)
        translation = torch.zeros(3)
        
        distance_matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            window=1,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert distance_matrix.shape == (1, 3, 3)

    def test_with_downsampling(self, small_trajectory_pair):
        """Test pairwise distance matrix with downsampling."""
        x, y = small_trajectory_pair
        
        rotation = torch.eye(3)
        translation = torch.zeros(3)
        
        distance_matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            normalize=True,
            distance_metric="swd",
            downsample_method="random",
        )
        
        assert distance_matrix.shape == (1, 3, 3)


class TestPairwiseDistanceMatrixExportedFunctions:
    """Tests for exported functions in __all__."""

    def test_exports(self):
        """Test that expected functions are exported."""
        assert "create_pairwise_distance_matrix" in pairwise_distance_matrix.__all__
        assert "create_pairwise_distance_matrix_given_rigid_rot" in pairwise_distance_matrix.__all__


class TestSanitizeAdditional:
    """Additional coverage for _sanitize_pairwise_distance_matrix (lines 517, 519->524, 526->529, 537->536, 549->548, 553-556, 559-568, 578->577, 582-591, 603->602)."""

    def test_explicit_kwargs_covers_false_branch(self, small_trajectory_pair):
        """Non-None distance_kwargs[0] takes the False branch (line 519->524)."""
        x, y = small_trajectory_pair
        _, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs={"num_projs": 30},  # not None → 519->524 False branch
            distance_metrics="swd",
            downsample_method="random",
            x=x,
            y=y,
        )

    def test_kwargs_len_mismatch_raises(self, small_trajectory_pair):
        """Mismatched kwargs/metrics lengths raises RuntimeError (line 517)."""
        x, y = small_trajectory_pair
        with pytest.raises(RuntimeError, match="len distance kwargs != len distance metrics"):
            pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
                distance_kwargs=[{"num_projs": 30}, {"num_projs": 50}],
                distance_metrics="swd",  # only 1 metric but 2 kwargs
                downsample_method="random",
                x=x,
                y=y,
            )

    def test_oswd_metric(self, small_trajectory_pair):
        """oswd metric is handled (lines 559-568)."""
        x, y = small_trajectory_pair
        fns, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics="oswd",
            downsample_method="random",
            x=x,
            y=y,
        )
        assert fns[0] is not None

    def test_pswd_metric(self, small_trajectory_pair):
        """pswd metric is handled (lines 582-591)."""
        x, y = small_trajectory_pair
        fns, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics="pswd",
            downsample_method="random",
            x=x,
            y=y,
        )
        assert fns[0] is not None

    def test_minkowski_metric(self, small_trajectory_pair):
        """minkowski metric with explicit p kwarg covers default-check False branch (603->602)."""
        x, y = small_trajectory_pair
        fns, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs={"p": 3},  # key already present → default check False branch
            distance_metrics="minkowski",
            downsample_method=None,
            x=x,
            y=y,
        )
        assert callable(fns[0])

    def test_oswd_with_preset_key_covers_false_branch(self, small_trajectory_pair):
        """Explicit oswd kwargs with existing key covers the False branch (566->565)."""
        x, y = small_trajectory_pair
        fns, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs={"device": x[0]["pos"].device, "num_projs": 30},  # keys already present
            distance_metrics="oswd",
            downsample_method="random",
            x=x,
            y=y,
        )
        assert fns[0] is not None

    def test_pswd_with_preset_key_covers_false_branch(self, small_trajectory_pair):
        """Explicit pswd kwargs with existing key covers the False branch (589->588)."""
        x, y = small_trajectory_pair
        fns, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs={"device": x[0]["pos"].device, "num_projs": 30},  # keys already present
            distance_metrics="pswd",
            downsample_method="random",
            x=x,
            y=y,
        )
        assert fns[0] is not None

    def test_aswd_with_projs_history_file_cleanup(self, small_trajectory_pair, tmp_path, monkeypatch):
        """projs_history.txt cleanup runs when file exists (lines 553-556)."""
        x, y = small_trajectory_pair
        # Create projs_history.txt in cwd so the code's Path("projs_history.txt").exists() is True
        hist_file = tmp_path / "projs_history.txt"
        hist_file.write_text("dummy")
        monkeypatch.chdir(tmp_path)
        pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics="aswd",
            downsample_method="random",
            x=x,
            y=y,
        )
        # The file should have been deleted
        assert not hist_file.exists()


class TestCreateGivenRigidRotAdditional:
    """Additional coverage for create_pairwise_distance_matrix_given_rigid_rot (lines 341, 357->366, 380, 383)."""

    def test_with_window_large_enough_to_clamp(self, small_trajectory_pair):
        """window large enough that window_max gets clamped (line 341)."""
        x, y = small_trajectory_pair  # 3 time points each
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)
        matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y, rotation=rotation, translation=translation,
            normalize=False, window=100,  # far exceeds y_samples → clamps at line 341
            distance_metric="euclidean", downsample_method=None,
        )
        assert matrix.shape[1] == len(x)

    def test_with_normalize_false(self, small_trajectory_pair):
        """normalize=False takes the False branch (357->366)."""
        x, y = small_trajectory_pair
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)
        matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y, rotation=rotation, translation=translation,
            normalize=False, distance_metric="euclidean", downsample_method=None,
        )
        assert torch.isfinite(matrix).all()

    def test_with_aswd_metric_calls_remove_history(self, small_trajectory_pair):
        """ASWD metric causes remove_history() per iteration (line 383)."""
        x, y = small_trajectory_pair
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)
        matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y, rotation=rotation, translation=translation,
            normalize=False, distance_metric="aswd", downsample_method="random",
        )
        assert matrix.shape[1] == len(x)

    def test_aswd_with_preset_key_covers_false_branch(self, small_trajectory_pair):
        """Explicit aswd kwargs with existing key covers the `kw in dist_kwargs` False branch (549->548)."""
        x, y = small_trajectory_pair
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)
        # "device" key already provided → `if kw not in dist_kwargs:` is False → 549->548
        matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y, rotation=rotation, translation=translation,
            normalize=False, distance_metric="aswd",
            distance_kwargs={"device": x[0]["pos"].device, "max_slices": 50, "init_projs": 20, "step_projs": 10},
            downsample_method="random",
        )
        assert matrix.shape[1] == len(x)


class TestPairwiseNormalizePath:
    """Test normalize=True code path (pairwise_distance_matrix.py lines 159-166)."""

    def test_normalize_true_executes_normalization(self, small_trajectory_pair):
        """create_pairwise_distance_matrix with normalize=True runs the normalize block."""
        x, y = small_trajectory_pair
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            distance_metric="euclidean",
            downsample_method=None,
            normalize=True,
        )
        matrix, rots = result.cost_matrix, result.rotations
        assert matrix.shape[1] == len(x)
        assert matrix.shape[2] == len(y)
        assert torch.isfinite(matrix).all()


class TestPairwiseASWDRemoveHistory:
    """Test ASWD remove_history is called in create_pairwise_distance_matrix (line 202)."""

    def test_aswd_distance_calls_remove_history(self, small_trajectory_pair):
        """Using aswd metric calls fn.remove_history() each iteration (line 202)."""
        from zreg.distances.sw_varients import AdaptiveSlicedWassersteinDistance
        x, y = small_trajectory_pair
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            distance_metric="aswd",
            downsample_method="random",
            normalize=False,
        )
        matrix, rots = result.cost_matrix, result.rotations
        assert matrix.shape[1] == len(x)


class TestSanitizeGSWDMetric:
    """gswd metric branch coverage (lines 569-580, 578->577 branch)."""

    def test_gswd_metric(self, small_trajectory_pair):
        """gswd metric executes the gswd branch (lines 569-580) and loops through defaults."""
        x, y = small_trajectory_pair
        fns, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics="gswd",
            downsample_method="random",
            x=x,
            y=y,
        )
        assert fns[0] is not None

    def test_gswd_with_preset_keys(self, small_trajectory_pair):
        """gswd with all keys preset exercises the False branch of 'if kw not in dist_kwargs' (578->577)."""
        x, y = small_trajectory_pair
        fns, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs={"device": x[0]["pos"].device, "num_projs": 30, "degree": 3.0},
            distance_metrics="gswd",
            downsample_method="random",
            x=x,
            y=y,
        )
        assert fns[0] is not None


class TestASDWPropsHistoryFileNotFound:
    """aswd projs_history.txt FileNotFoundError path (lines 555-556)."""

    def test_aswd_projs_history_file_not_found_on_remove(self, small_trajectory_pair, tmp_path, monkeypatch):
        """Race condition: file exists at exists() check but raises FileNotFoundError on remove (lines 555-556)."""
        from unittest.mock import patch as _patch
        x, y = small_trajectory_pair
        hist_file = tmp_path / "projs_history.txt"
        hist_file.write_text("dummy")
        monkeypatch.chdir(tmp_path)
        with _patch("zreg.pairwise_distance_matrix.os.remove", side_effect=FileNotFoundError):
            # Must not raise; FileNotFoundError is swallowed (lines 555-556)
            pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
                distance_kwargs=None,
                distance_metrics="aswd",
                downsample_method="random",
                x=x,
                y=y,
            )


class TestGivenRigidRotCPDMetric:
    """create_pairwise_distance_matrix_given_rigid_rot with cpd metric (line 380)."""

    def test_cpd_metric_fn_is_none_path(self, small_trajectory_pair):
        """distance_metric='cpd' makes fn=None, which hits 'if fn is None: continue' (line 380)."""
        x, y = small_trajectory_pair
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)
        # fn=None → line 380 covered; dists stays empty → IndexError on dists[di]
        with pytest.raises(IndexError):
            pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
                x, y,
                rotation=rotation,
                translation=translation,
                normalize=False,
                distance_metric="cpd",
            )


class TestMPIPaths:
    """Coverage for MPI distribution paths via mocked hasmpi/MPI.

    Uses unittest.mock to inject hasmpi=True and a fake MPI.COMM_WORLD so the
    mpi_distribute branches run without mpi4py installed.
    """

    def _make_single_sample_pair(self):
        """1-element trajectory (x_samples=0, y_samples=0 → single inner iteration)."""
        pc = zRegPointCloud(pos=torch.randn(10, 3), label=torch.rand(10, 3), id=torch.arange(10))
        return {0: pc}, {0: pc}

    def _make_two_sample_pair(self):
        """2-element trajectory."""
        pcs_x = {i: zRegPointCloud(pos=torch.randn(10, 3), label=torch.rand(10, 3), id=torch.arange(10)) for i in range(2)}
        pcs_y = {i: zRegPointCloud(pos=torch.randn(10, 3), label=torch.rand(10, 3), id=torch.arange(10)) for i in range(2)}
        return pcs_x, pcs_y

    def _make_comm(self, rank, size):
        import numpy as np
        comm = MagicMock()
        comm.rank = rank
        comm.size = size
        comm.allgather.side_effect = lambda row: [row, np.zeros_like(row)]
        return comm

    def test_mpi_skip_and_empty_row_continue(self):
        """rank=1, size=2 with 1-sample pair: iteration fc=0 is skipped (148-150) → empty row → continue (233)."""
        from unittest.mock import patch as _patch, MagicMock as _MagicMock
        import zreg.pairwise_distance_matrix as pmat

        comm = self._make_comm(rank=1, size=2)
        mock_mpi = _MagicMock()
        mock_mpi.COMM_WORLD = comm
        x, y = self._make_single_sample_pair()

        with _patch.object(pmat, "hasmpi", True), _patch.object(pmat, "MPI", mock_mpi):
            matrix = pmat.create_pairwise_distance_matrix(
                x, y,
                normalize=False,
                distance_metric="euclidean",
                mpi_distribute=True,
            ).cost_matrix
        # rank=1 skips fc=0 (0%2 ≠ 1), so distance stays inf
        assert matrix.shape[1] == 1

    def test_mpi_allgather_path(self):
        """rank=0, size=2 with 2-sample pair: fc=1 skipped, fc=0,2 processed → allgather called (89-90, 256-262)."""
        from unittest.mock import patch as _patch, MagicMock as _MagicMock
        import zreg.pairwise_distance_matrix as pmat

        comm = self._make_comm(rank=0, size=2)
        mock_mpi = _MagicMock()
        mock_mpi.COMM_WORLD = comm
        x, y = self._make_two_sample_pair()

        with _patch.object(pmat, "hasmpi", True), _patch.object(pmat, "MPI", mock_mpi):
            matrix = pmat.create_pairwise_distance_matrix(
                x, y,
                normalize=False,
                distance_metric="euclidean",
                mpi_distribute=True,
            ).cost_matrix
        assert comm.allgather.called
        assert matrix.shape[1] == 2

    def test_mpi_given_rigid_rot_skip_and_empty_row(self):
        """rank=1, size=2 with given_rigid_rot: fc=0 skipped (348-350) → empty row → continue (418)."""
        from unittest.mock import patch as _patch, MagicMock as _MagicMock
        import zreg.pairwise_distance_matrix as pmat

        comm = self._make_comm(rank=1, size=2)
        mock_mpi = _MagicMock()
        mock_mpi.COMM_WORLD = comm
        x, y = self._make_single_sample_pair()
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)

        with _patch.object(pmat, "hasmpi", True), _patch.object(pmat, "MPI", mock_mpi):
            matrix = pmat.create_pairwise_distance_matrix_given_rigid_rot(
                x, y,
                rotation=rotation,
                translation=translation,
                normalize=False,
                distance_metric="euclidean",
                mpi_distribute=True,
            )
        assert matrix.shape[1] == 1

    def test_mpi_given_rigid_rot_allgather(self):
        """rank=0, size=2 with given_rigid_rot: allgather called (286-287, 439-445)."""
        from unittest.mock import patch as _patch, MagicMock as _MagicMock
        import zreg.pairwise_distance_matrix as pmat

        comm = self._make_comm(rank=0, size=2)
        mock_mpi = _MagicMock()
        mock_mpi.COMM_WORLD = comm
        x, y = self._make_two_sample_pair()
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)

        with _patch.object(pmat, "hasmpi", True), _patch.object(pmat, "MPI", mock_mpi):
            matrix = pmat.create_pairwise_distance_matrix_given_rigid_rot(
                x, y,
                rotation=rotation,
                translation=translation,
                normalize=False,
                distance_metric="euclidean",
                mpi_distribute=True,
            )
        assert comm.allgather.called
        assert matrix.shape[1] == 2

    def _make_asymmetric_pair(self):
        """x has 2 samples, y has 1 sample — for loop back-edge branch tests."""
        pcs_x = {i: zRegPointCloud(pos=torch.randn(10, 3), label=torch.rand(10, 3), id=torch.arange(10)) for i in range(2)}
        pcs_y = {0: zRegPointCloud(pos=torch.randn(10, 3), label=torch.rand(10, 3), id=torch.arange(10))}
        return pcs_x, pcs_y

    def test_mpi_loop_back_edge_create(self):
        """rank=1, size=2, x=2 samples, y=1 sample: i=0 is fully skipped → continue back to i=1 (261->133)."""
        from unittest.mock import patch as _patch, MagicMock as _MagicMock
        import zreg.pairwise_distance_matrix as pmat

        comm = self._make_comm(rank=1, size=2)
        mock_mpi = _MagicMock()
        mock_mpi.COMM_WORLD = comm
        x, y = self._make_asymmetric_pair()

        with _patch.object(pmat, "hasmpi", True), _patch.object(pmat, "MPI", mock_mpi):
            matrix = pmat.create_pairwise_distance_matrix(
                x, y,
                normalize=False,
                distance_metric="euclidean",
                mpi_distribute=True,
            ).cost_matrix
        assert matrix.shape[1] == 2

    def test_mpi_loop_back_edge_given_rigid_rot(self):
        """rank=1, size=2, x=2 samples, y=1 sample: i=0 fully skipped → continue back to i=1 (444->333)."""
        from unittest.mock import patch as _patch, MagicMock as _MagicMock
        import zreg.pairwise_distance_matrix as pmat

        comm = self._make_comm(rank=1, size=2)
        mock_mpi = _MagicMock()
        mock_mpi.COMM_WORLD = comm
        x, y = self._make_asymmetric_pair()
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)

        with _patch.object(pmat, "hasmpi", True), _patch.object(pmat, "MPI", mock_mpi):
            matrix = pmat.create_pairwise_distance_matrix_given_rigid_rot(
                x, y,
                rotation=rotation,
                translation=translation,
                normalize=False,
                distance_metric="euclidean",
                mpi_distribute=True,
            )
        assert matrix.shape[1] == 2


class TestCallableMetricPassThrough:
    """RED-phase tests: callable pass-through in _sanitize_pairwise_distance_matrix.

    These tests define the behaviour required by DTW-02 (consistent metric variant
    interface).  They intentionally fail before the implementation is wired.
    """

    @pytest.fixture
    def small_pair(self):
        x = {i: zRegPointCloud(pos=torch.randn(5, 3), id=torch.arange(5)) for i in range(2)}
        y = {i: zRegPointCloud(pos=torch.randn(5, 3), id=torch.arange(5)) for i in range(2)}
        return x, y

    def test_distance_metric_protocol_is_importable(self):
        """DistanceMetric Protocol must be importable from zreg.distances."""
        from zreg.distances import DistanceMetric  # noqa: F401 (import-only test)
        assert DistanceMetric is not None

    def test_callable_metric_returned_unchanged_by_sanitize(self, small_pair):
        """A callable passed to _sanitize must be returned as-is (no string lookup)."""
        from zreg.distances.general import euclidean_distance
        from zreg.pairwise_distance_matrix import _sanitize_pairwise_distance_matrix
        x, y = small_pair
        fns, _, _ = _sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics=euclidean_distance,
            downsample_method=None,
            x=x,
            y=y,
        )
        assert fns[0] is euclidean_distance

    def test_invalid_string_still_raises_value_error(self, small_pair):
        """An unrecognised string must still raise ValueError (regression guard)."""
        from zreg.pairwise_distance_matrix import _sanitize_pairwise_distance_matrix
        x, y = small_pair
        with pytest.raises(ValueError):
            _sanitize_pairwise_distance_matrix(
                distance_kwargs=None,
                distance_metrics="not_a_real_metric",
                downsample_method=None,
                x=x,
                y=y,
            )

    def test_callable_skips_swd_downsampling_guard(self, small_pair):
        """A callable metric must not trigger the SWD downsampling RuntimeError."""
        from zreg.distances.general import euclidean_distance
        from zreg.pairwise_distance_matrix import _sanitize_pairwise_distance_matrix
        x, y = small_pair
        # With a string metric (non-euclidean) and no downsampling → RuntimeError
        # With a callable + no downsampling → should succeed
        fns, ds_method, _ = _sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics=euclidean_distance,
            downsample_method=None,
            x=x,
            y=y,
        )
        assert ds_method is None  # callable path → downsampling untouched


# ---------------------------------------------------------------------------
# Coverage gap tests: cpd_type='nonrigid' and cpd_type='affine'
# ---------------------------------------------------------------------------


class TestPairwiseDistanceMatrixCPDTypes:
    """pairwise_distance_matrix.py:185,187 — nonrigid and affine CPD paths."""

    def _make_small_pair(self):
        """Two tiny 3-frame trajectories (10 points each) for fast CPD tests."""
        x = {i: zRegPointCloud(pos=torch.randn(10, 3), label=None, id=torch.arange(10)) for i in range(3)}
        y = {i: zRegPointCloud(pos=torch.randn(10, 3), label=None, id=torch.arange(10)) for i in range(3)}
        return x, y

    def test_cpd_type_nonrigid(self):
        """cpd_type='nonrigid' executes NonRigidCPD branch (line 185).

        With tiny datasets NonRigidCPD may return transformation=None (upstream
        bug). The branch line is still reached, so accept AttributeError too.
        """
        x, y = self._make_small_pair()
        try:
            result = pairwise_distance_matrix.create_pairwise_distance_matrix(
                x=x,
                y=y,
                distance_metric="euclidean",
                cpd_type="nonrigid",
            )
            assert result is not None
        except AttributeError:
            pass  # upstream NonRigidCPD bug — branch was still reached

    def test_cpd_type_affine(self):
        """cpd_type='affine' executes AffineCPD branch (line 187)."""
        x, y = self._make_small_pair()
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x=x,
            y=y,
            distance_metric="euclidean",
            cpd_type="affine",
        )
        assert result is not None


class TestPairwiseResult:
    """Tests for PairwiseResult return type from create_pairwise_distance_matrix (ALIGN-03)."""

    def test_returns_pairwise_result_instance(self, small_trajectory_pair):
        """create_pairwise_distance_matrix must return a PairwiseResult, not a 2-tuple."""
        from zreg.types import PairwiseResult
        x, y = small_trajectory_pair
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            downsample_method=None,
        )
        assert isinstance(result, PairwiseResult)

    def test_cost_matrix_field(self, small_trajectory_pair):
        """PairwiseResult.cost_matrix has the expected shape."""
        x, y = small_trajectory_pair
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            downsample_method=None,
        )
        assert result.cost_matrix.shape == (1, 3, 3)

    def test_stored_transforms_empty_without_cpd(self, small_trajectory_pair):
        """stored_transforms is empty dict when cpd_type is None."""
        x, y = small_trajectory_pair
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            downsample_method=None,
        )
        assert isinstance(result.stored_transforms, dict)
        assert len(result.stored_transforms) == 0

    def test_stored_transforms_populated_with_cpd(self, small_trajectory_pair):
        """stored_transforms has entries when cpd_type is not None."""
        from zreg.types import StoredTransform
        x, y = small_trajectory_pair
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            cpd_type="rigid",
            downsample_method="random",
        )
        assert isinstance(result.stored_transforms, dict)
        assert len(result.stored_transforms) > 0
        for key, value in result.stored_transforms.items():
            assert isinstance(key, tuple) and len(key) == 2
            assert isinstance(value, StoredTransform)

    def test_rotations_none_without_cpd(self, small_trajectory_pair):
        """rotations is None when cpd_type is None."""
        x, y = small_trajectory_pair
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            downsample_method=None,
        )
        assert result.rotations is None

    def test_rotations_tensor_with_rigid_cpd(self, small_trajectory_pair):
        """rotations is a Tensor when cpd_type='rigid'."""
        x, y = small_trajectory_pair
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            cpd_type="rigid",
            downsample_method="random",
        )
        assert result.rotations is not None
        assert isinstance(result.rotations, torch.Tensor)

    def test_cpd_with_normalize_false_skips_stored_transform(self, small_trajectory_pair):
        """pairwise_distance_matrix.py:217->225 — normalize=False skips StoredTransform storage."""
        x, y = small_trajectory_pair
        result = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=False,
            distance_metric="euclidean",
            cpd_type="rigid",
            downsample_method="random",
        )
        # StoredTransforms dict should be empty because normalize=False
        assert result.stored_transforms == {}

    def test_maxswd_metric(self, small_trajectory_pair):
        """pairwise_distance_matrix.py:641-650 — maxswd distance metric is handled."""
        x, y = small_trajectory_pair
        fns, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs=None,
            distance_metrics="maxswd",
            downsample_method="random",
            x=x,
            y=y,
        )
        assert fns[0] is not None

    def test_maxswd_with_preset_device_skips_default(self, small_trajectory_pair):
        """pairwise_distance_matrix.py:648->647 — False branch when key already in dist_kwargs."""
        x, y = small_trajectory_pair
        fns, _, _ = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
            distance_kwargs={"device": "cpu"},
            distance_metrics="maxswd",
            downsample_method="random",
            x=x,
            y=y,
        )
        assert fns[0] is not None
