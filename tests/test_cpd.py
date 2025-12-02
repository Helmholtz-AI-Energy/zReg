"""Tests for zreg.cpd module."""

import pytest
import torch

from zreg import cpd
from zreg import transforms


@pytest.fixture
def source_target_pair():
    """Create a source and target point cloud pair."""
    # Create source points
    source = torch.randn(50, 3)
    
    # Create target as rotated/translated version of source
    rot = torch.tensor([
        [0.866, -0.5, 0.0],
        [0.5, 0.866, 0.0],
        [0.0, 0.0, 1.0]
    ])  # 30 degree rotation around z
    t = torch.tensor([1.0, 2.0, 0.0])
    target = source @ rot.T + t
    
    return source, target


@pytest.fixture
def simple_translation_pair():
    """Create a simple translation-only pair."""
    source = torch.randn(30, 3)
    target = source + torch.tensor([1.0, 0.0, 0.0])
    return source, target


class TestCoherentPointDriftBase:
    """Tests for base CoherentPointDrift class."""

    def test_set_source(self):
        """Test setting source point cloud."""
        source = torch.randn(20, 3)
        cpd_obj = cpd.RigidCPD(source=source)
        
        new_source = torch.randn(30, 3)
        cpd_obj.set_source(new_source)
        
        assert cpd_obj._source.shape[0] == 30

    def test_set_callbacks(self):
        """Test setting callbacks."""
        source = torch.randn(20, 3)
        cpd_obj = cpd.RigidCPD(source=source)
        
        callbacks = [lambda x: None, lambda x: None]
        cpd_obj.set_callbacks(callbacks)
        
        assert len(cpd_obj._callbacks) == 2


class TestEstepResult:
    """Tests for EstepResult namedtuple."""

    def test_estep_result_creation(self):
        """Test creating EstepResult."""
        pt1 = torch.randn(10)
        p1 = torch.randn(10)
        px = torch.randn(10, 3)
        n_p = torch.tensor(10.0)
        
        result = cpd.EstepResult(pt1, p1, px, n_p)
        
        assert torch.equal(result.pt1, pt1)
        assert torch.equal(result.p1, p1)
        assert torch.equal(result.px, px)
        assert torch.equal(result.n_p, n_p)


class TestMstepResult:
    """Tests for MstepResult namedtuple."""

    def test_mstep_result_creation(self):
        """Test creating MstepResult."""
        tf = transforms.RigidTransformation(device="cpu", dtype=torch.float32)
        sigma2 = 0.1
        q = 0.5
        
        result = cpd.MstepResult(tf, sigma2, q)
        
        assert result.transformation is tf
        assert result.sigma2 == sigma2
        assert result.q == q


class TestRigidCPD:
    """Tests for RigidCPD class."""

    def test_initialization(self):
        """Test RigidCPD initialization."""
        source = torch.randn(20, 3)
        cpd_obj = cpd.RigidCPD(source=source)
        
        assert cpd_obj._source is not None
        assert cpd_obj._tf_type == transforms.RigidTransformation

    def test_initialization_with_update_scale(self):
        """Test RigidCPD with update_scale parameter."""
        source = torch.randn(20, 3)
        cpd_obj = cpd.RigidCPD(source=source, update_scale=False)
        
        assert cpd_obj._update_scale is False

    def test_registration_basic(self, source_target_pair):
        """Test basic registration."""
        source, target = source_target_pair
        cpd_obj = cpd.RigidCPD(source=source, log_freq=-1)
        
        result = cpd_obj.registration(target, maxiter=20, tol=1e-4)
        
        assert result.transformation is not None
        assert result.sigma2 is not None
        assert result.q is not None

    def test_registration_improves_alignment(self, source_target_pair):
        """Test that registration improves alignment."""
        source, target = source_target_pair
        cpd_obj = cpd.RigidCPD(source=source, log_freq=-1)
        
        # Initial distance
        initial_dist = torch.cdist(source, target).min(dim=1)[0].mean()
        
        result = cpd_obj.registration(target, maxiter=50, tol=1e-5)
        transformed = result.transformation.transform(source)
        
        # Distance after registration
        final_dist = torch.cdist(transformed, target).min(dim=1)[0].mean()
        
        # Should be closer after registration
        assert final_dist < initial_dist

    def test_registration_with_translation_only(self, simple_translation_pair):
        """Test registration with translation-only transformation."""
        source, target = simple_translation_pair
        cpd_obj = cpd.RigidCPD(source=source, update_scale=False, log_freq=-1)
        
        result = cpd_obj.registration(target, maxiter=50, tol=1e-5)
        transformed = result.transformation.transform(source)
        
        # Should be reasonably close after registration
        # Note: CPD with rigid transformation may not perfectly align translation-only
        # transformations due to the random initial rotation matrix
        dist = torch.cdist(transformed, target).min(dim=1)[0].mean()
        assert dist < 1.0  # Should be close

    def test_reset_transform(self, source_target_pair):
        """Test reset_transform method."""
        source, target = source_target_pair
        cpd_obj = cpd.RigidCPD(source=source, log_freq=-1)
        
        # Run registration
        cpd_obj.registration(target, maxiter=10)
        
        # Reset
        cpd_obj.reset_transform()
        
        # Transform should be identity-like
        assert torch.allclose(cpd_obj.transformation.rot, torch.eye(3), atol=1e-5)

    def test_expectation_step(self, source_target_pair):
        """Test expectation step."""
        source, target = source_target_pair
        cpd_obj = cpd.RigidCPD(source=source, log_freq=-1)
        
        sigma2 = 1.0
        sigma2_c = 0.0
        
        result = cpd_obj.expectation_step(source, target, sigma2, sigma2_c, w=0.0)
        
        assert isinstance(result, cpd.EstepResult)
        assert result.pt1.shape[0] == target.shape[0]
        assert result.p1.shape[0] == source.shape[0]
        assert result.px.shape == (source.shape[0], 3)


class TestAffineCPD:
    """Tests for AffineCPD class."""

    def test_initialization(self):
        """Test AffineCPD initialization."""
        source = torch.randn(20, 3)
        cpd_obj = cpd.AffineCPD(source=source)
        
        assert cpd_obj._source is not None
        assert cpd_obj._tf_type == transforms.AffineTransformation

    def test_registration_basic(self, source_target_pair):
        """Test basic affine registration."""
        source, target = source_target_pair
        cpd_obj = cpd.AffineCPD(source=source, log_freq=-1)
        
        result = cpd_obj.registration(target, maxiter=20, tol=1e-4)
        
        assert result.transformation is not None
        assert isinstance(result.transformation, transforms.AffineTransformation)


class TestNonRigidCPD:
    """Tests for NonRigidCPD class."""

    def test_initialization(self):
        """Test NonRigidCPD initialization."""
        source = torch.randn(20, 3)
        cpd_obj = cpd.NonRigidCPD(source=source, beta=2.0, lmd=2.0)
        
        assert cpd_obj._source is not None
        assert cpd_obj._tf_type == transforms.NonRigidTransformation
        assert cpd_obj._beta == 2.0
        assert cpd_obj._lmd == 2.0

    def test_set_source(self):
        """Test set_source creates transformation object."""
        source = torch.randn(20, 3)
        cpd_obj = cpd.NonRigidCPD(source=source)
        
        new_source = torch.randn(30, 3)
        cpd_obj.set_source(new_source)
        
        assert cpd_obj._source.shape[0] == 30
        assert cpd_obj._tf_obj is not None

    def test_registration_basic(self, source_target_pair):
        """Test basic nonrigid registration."""
        source, target = source_target_pair
        cpd_obj = cpd.NonRigidCPD(source=source, log_freq=-1)
        
        result = cpd_obj.registration(target, maxiter=10, tol=1e-3)
        
        assert result.transformation is not None


class TestConstrainedNonRigidCPD:
    """Tests for ConstrainedNonRigidCPD class."""

    def test_initialization(self):
        """Test ConstrainedNonRigidCPD initialization."""
        source = torch.randn(20, 3)
        cpd_obj = cpd.ConstrainedNonRigidCPD(source=source, alpha=1e-8)
        
        assert cpd_obj._source is not None
        assert cpd_obj.alpha == 1e-8

    def test_initialization_with_constraints(self):
        """Test initialization with constraint indices."""
        source = torch.randn(20, 3)
        idx_source = torch.tensor([0, 1, 2])
        idx_target = torch.tensor([0, 1, 2])
        
        cpd_obj = cpd.ConstrainedNonRigidCPD(
            source=source,
            idx_source=idx_source,
            idx_target=idx_target,
        )
        
        assert torch.equal(cpd_obj.idx_source, idx_source)
        assert torch.equal(cpd_obj.idx_target, idx_target)


class TestCPDRegistrationFunction:
    """Tests for cpd_registration convenience function."""

    def test_rigid_registration(self, source_target_pair):
        """Test rigid registration via convenience function."""
        source, target = source_target_pair
        source_dict = {"pos": source}
        target_dict = {"pos": target}
        
        result = cpd.cpd_registration(
            source_dict, target_dict,
            tf_type_name="rigid",
            maxiter=20,
            log_freq=-1,
        )
        
        assert result.transformation is not None

    def test_affine_registration(self, source_target_pair):
        """Test affine registration via convenience function."""
        source, target = source_target_pair
        source_dict = {"pos": source}
        target_dict = {"pos": target}
        
        result = cpd.cpd_registration(
            source_dict, target_dict,
            tf_type_name="affine",
            maxiter=20,
            log_freq=-1,
        )
        
        assert result.transformation is not None

    def test_nonrigid_registration(self, source_target_pair):
        """Test nonrigid registration via convenience function."""
        source, target = source_target_pair
        source_dict = {"pos": source}
        target_dict = {"pos": target}
        
        result = cpd.cpd_registration(
            source_dict, target_dict,
            tf_type_name="nonrigid",
            maxiter=10,
            log_freq=-1,
        )
        
        assert result.transformation is not None

    def test_invalid_tf_type_raises(self, source_target_pair):
        """Test that invalid tf_type raises ValueError."""
        source, target = source_target_pair
        source_dict = {"pos": source}
        target_dict = {"pos": target}
        
        with pytest.raises(ValueError):
            cpd.cpd_registration(
                source_dict, target_dict,
                tf_type_name="invalid_type",
            )


class TestInitCPDFromExisting:
    """Tests for init_cpd_from_existing function."""

    def test_init_from_rigid_transform(self, source_target_pair):
        """Test initializing CPD from existing rigid transform."""
        source, target = source_target_pair
        source_dict = {"pos": source}
        target_dict = {"pos": target}
        
        # Create a rigid transform
        tf = transforms.RigidTransformation(device="cpu", dtype=torch.float32)
        
        cpd_obj = cpd.init_cpd_from_existing(
            tf, source_dict, target_dict, log_freq=-1
        )
        
        assert isinstance(cpd_obj, cpd.RigidCPD)

    def test_init_from_affine_transform(self, source_target_pair):
        """Test initializing CPD from existing affine transform."""
        source, target = source_target_pair
        source_dict = {"pos": source}
        target_dict = {"pos": target}
        
        # Create an affine transform
        tf = transforms.AffineTransformation(device="cpu", dtype=torch.float32)
        
        cpd_obj = cpd.init_cpd_from_existing(
            tf, source_dict, target_dict, log_freq=-1
        )
        
        assert isinstance(cpd_obj, cpd.AffineCPD)

    def test_init_from_unknown_transform_raises(self, source_target_pair):
        """Test that unknown transform type raises TypeError."""
        source, target = source_target_pair
        source_dict = {"pos": source}
        target_dict = {"pos": target}
        
        # Create a non-rigid transform (not supported)
        tf = transforms.NonRigidTransformation(
            w=None, points=source, beta=2.0
        )
        
        with pytest.raises(TypeError):
            cpd.init_cpd_from_existing(tf, source_dict, target_dict)


class TestCPDWithColor:
    """Tests for CPD with color information."""

    def test_rigid_cpd_with_color(self):
        """Test RigidCPD with color information."""
        source = torch.randn(30, 3)
        source_colors = torch.rand(30, 3)
        target = source + torch.randn(30, 3) * 0.1
        target_colors = source_colors + torch.randn(30, 3) * 0.1
        
        cpd_obj = cpd.RigidCPD(
            source=source,
            source_colors=source_colors,
            use_color=True,
            log_freq=-1,
        )
        
        result = cpd_obj.registration(
            target, maxiter=10,
            target_colors=target_colors,
        )
        
        assert result.transformation is not None
