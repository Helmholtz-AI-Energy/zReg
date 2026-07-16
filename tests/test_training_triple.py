"""Tests for `eval.types.TrainingTriple` (Phase 46 Plan 02, D-02).

Covers construction, frozen-immutability (matching AlignResult/LabelResult
conventions), and label-accessor correctness — the accessors MUST read
`pc["label"]`, never `pc["id"]` (45-DESIGN.md "label vs id Discipline").
"""

# zreg (and scipy) must be imported before torch on macOS ARM to avoid
# duplicate libomp initialisation (SIGABRT). No open3d dependency.
from zreg.dataset import zRegPointCloud

import pytest
import torch
from pydantic import ValidationError

from eval.types import TrainingTriple


@pytest.fixture
def source_cloud():  # pragma: no cover
    return zRegPointCloud(
        pos=torch.randn(20, 3),
        label=torch.randint(0, 6, (20,)),
        id=torch.arange(20),
    )


@pytest.fixture
def target_cloud():  # pragma: no cover
    return zRegPointCloud(
        pos=torch.randn(20, 3),
        label=torch.randint(0, 6, (20,)),
        id=torch.arange(20),
    )


class TestTrainingTripleConstruction:
    def test_constructs_successfully(self, source_cloud, target_cloud):
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        assert isinstance(triple, TrainingTriple)

    def test_seed_field(self, source_cloud, target_cloud):
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        assert triple.seed == 5

    def test_source_cloud_identity(self, source_cloud, target_cloud):
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        assert triple.source_cloud is source_cloud

    def test_target_cloud_identity(self, source_cloud, target_cloud):
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        assert triple.target_cloud is target_cloud


class TestTrainingTripleLabelAccessors:
    def test_source_labels_matches_source_cloud_label(self, source_cloud, target_cloud):
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        assert torch.equal(triple.source_labels, source_cloud["label"])

    def test_target_labels_matches_target_cloud_label(self, source_cloud, target_cloud):
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        assert torch.equal(triple.target_labels, target_cloud["label"])

    def test_source_labels_not_id(self, source_cloud, target_cloud):
        """Label-vs-id discipline (45-DESIGN.md): accessor must never read pc['id']."""
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        assert not torch.equal(triple.source_labels, source_cloud["id"])

    def test_target_labels_not_id(self, source_cloud, target_cloud):
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        assert not torch.equal(triple.target_labels, target_cloud["id"])


class TestTrainingTripleFrozen:
    def test_seed_reassignment_raises(self, source_cloud, target_cloud):
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        with pytest.raises(ValidationError):
            triple.seed = 9

    def test_source_cloud_reassignment_raises(self, source_cloud, target_cloud):
        triple = TrainingTriple(source_cloud=source_cloud, target_cloud=target_cloud, seed=5)
        with pytest.raises(ValidationError):
            triple.source_cloud = target_cloud


class TestTrainingTripleImportable:
    def test_importable_from_eval_types(self):
        from eval.types import TrainingTriple as ImportedTrainingTriple
        assert ImportedTrainingTriple is TrainingTriple
