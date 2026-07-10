"""Tests for eval.config.EvalConfig.

Tests cover configuration loading, validation, and field defaults.
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from pydantic import ValidationError

from eval.config import AlignmentPreprocessingConfig, EvalConfig, EvalConfigError


class TestEvalConfigBasics:
    """Basic EvalConfig instantiation and defaults."""

    def test_eval_config_minimal_instantiation(self, tmp_path):
        """Test that EvalConfig can be instantiated with only data_path."""
        config = EvalConfig(data_path=str(tmp_path / "data.mat"))
        assert config.data_path == str(tmp_path / "data.mat")
        assert config.data_format == "tracklets"
        assert config.n_synthetic == 100
        assert config.run_alignment is True
        assert config.alignment_method == "cpd"

    def test_eval_config_alignment_method_default_is_cpd(self, tmp_path):
        """Test that alignment_method defaults to 'cpd'."""
        config = EvalConfig(data_path=str(tmp_path / "data.mat"))
        assert config.alignment_method == "cpd"

    def test_eval_config_alignment_method_explicit_cpd(self, tmp_path):
        """Test that alignment_method='cpd' can be set explicitly."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="cpd",
        )
        assert config.alignment_method == "cpd"

    def test_eval_config_alignment_method_explicit_icp(self, tmp_path):
        """Test that alignment_method='icp' can be set explicitly."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="icp",
        )
        assert config.alignment_method == "icp"


class TestEvalConfigAlignmentMethodValidation:
    """Tests for alignment_method field validation."""

    def test_alignment_method_invalid_value_raises(self, tmp_path):
        """Test that invalid alignment_method raises ValueError."""
        with pytest.raises(ValueError, match="alignment_method must be 'cpd', 'icp', or 'swd'"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="invalid_method",
            )

    def test_alignment_method_case_sensitive(self, tmp_path):
        """Test that alignment_method validation is case-sensitive."""
        with pytest.raises(ValueError, match="alignment_method must be 'cpd', 'icp', or 'swd'"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="ICP",  # uppercase should fail
            )

    def test_alignment_method_empty_string_raises(self, tmp_path):
        """Test that empty string for alignment_method raises ValueError."""
        with pytest.raises(ValueError, match="alignment_method must be 'cpd', 'icp', or 'swd'"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="",
            )

    def test_alignment_method_none_raises(self, tmp_path):
        """Test that None for alignment_method raises error (type mismatch)."""
        with pytest.raises(Exception):  # pydantic will raise on type mismatch
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method=None,
            )


class TestEvalConfigLabelTransferMethodValidation:
    """Tests for label_transfer_method field validation."""

    def test_label_transfer_method_invalid_value_raises(self, tmp_path):
        """Test that invalid label_transfer_method raises ValueError."""
        with pytest.raises(
            ValueError, match="label_transfer_method must be 'knn_voting' or 'cpd_weighted'"
        ):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                label_transfer_method="invalid_method",
            )

    def test_label_transfer_method_case_sensitive(self, tmp_path):
        """Test that label_transfer_method validation is case-sensitive."""
        with pytest.raises(
            ValueError, match="label_transfer_method must be 'knn_voting' or 'cpd_weighted'"
        ):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                label_transfer_method="KNN_VOTING",  # uppercase should fail
            )

    def test_label_transfer_method_empty_string_raises(self, tmp_path):
        """Test that empty string for label_transfer_method raises ValueError."""
        with pytest.raises(
            ValueError, match="label_transfer_method must be 'knn_voting' or 'cpd_weighted'"
        ):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                label_transfer_method="",
            )

    def test_label_transfer_method_default_is_knn_voting(self, tmp_path):
        """Test that label_transfer_method defaults to 'knn_voting'."""
        config = EvalConfig(data_path=str(tmp_path / "data.mat"))
        assert config.label_transfer_method == "knn_voting"

    def test_label_transfer_method_cpd_weighted_valid(self, tmp_path):
        """Test that label_transfer_method='cpd_weighted' can be set explicitly."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            label_transfer_method="cpd_weighted",
        )
        assert config.label_transfer_method == "cpd_weighted"


class TestEvalConfigYAMLLoading:
    """Tests for loading alignment_method from YAML files."""

    def test_load_yaml_with_alignment_method_cpd(self, tmp_path):
        """Test loading EvalConfig from YAML with alignment_method='cpd'."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
            "alignment_method": "cpd",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = EvalConfig.from_yaml(str(yaml_path))
        assert config.alignment_method == "cpd"

    def test_load_yaml_with_alignment_method_icp(self, tmp_path):
        """Test loading EvalConfig from YAML with alignment_method='icp'."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
            "alignment_method": "icp",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = EvalConfig.from_yaml(str(yaml_path))
        assert config.alignment_method == "icp"

    def test_load_yaml_without_alignment_method_defaults_to_cpd(self, tmp_path):
        """Test that YAML without alignment_method defaults to 'cpd'."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = EvalConfig.from_yaml(str(yaml_path))
        assert config.alignment_method == "cpd"

    def test_load_yaml_with_invalid_alignment_method_raises(self, tmp_path):
        """Test that YAML with invalid alignment_method raises EvalConfigError."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
            "alignment_method": "invalid",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        with pytest.raises(EvalConfigError, match="alignment_method"):
            EvalConfig.from_yaml(str(yaml_path))

    def test_load_yaml_with_multiple_fields_and_alignment_method(self, tmp_path):
        """Test loading YAML with multiple fields including alignment_method."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
            "n_synthetic": 200,
            "alignment_method": "icp",
            "run_alignment": False,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = EvalConfig.from_yaml(str(yaml_path))
        assert config.alignment_method == "icp"
        assert config.n_synthetic == 200
        assert config.run_alignment is False


class TestEvalConfigAlignmentMethodIntegration:
    """Integration tests: alignment_method with other EvalConfig fields."""

    def test_alignment_method_independent_of_pipeline_mode(self, tmp_path):
        """Test that alignment_method works with both pipeline_mode values."""
        # paired mode
        config_paired = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            pipeline_mode="paired",
            alignment_method="icp",
        )
        assert config_paired.alignment_method == "icp"
        assert config_paired.pipeline_mode == "paired"

        # synthetic mode
        config_synthetic = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            pipeline_mode="synthetic",
            alignment_method="cpd",
        )
        assert config_synthetic.alignment_method == "cpd"
        assert config_synthetic.pipeline_mode == "synthetic"

    def test_alignment_method_with_various_tiers(self, tmp_path):
        """Test that alignment_method works with all tier values."""
        for tier in ["sanity", "dev", "full"]:
            config = EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                tier=tier,
                alignment_method="icp",
            )
            assert config.alignment_method == "icp"
            assert config.tier == tier

    def test_alignment_method_preserves_other_defaults(self, tmp_path):
        """Test that setting alignment_method doesn't affect other defaults."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="icp",
        )
        # Verify other fields still have their defaults
        assert config.data_format == "tracklets"
        assert config.ground_truth_path is None
        assert config.n_synthetic == 100
        assert config.transform_degree == 0.1
        assert config.run_label_transfer is True
        assert config.val_split == 0.2
        assert config.save_plots is True
        assert config.verbose is False
        assert config.output_dir == "experiments/runs"


class TestAlignmentPreprocessingConfig:
    """Phase 41: AlignmentPreprocessingConfig validation and EvalConfig integration (ALIGN-06-03)."""

    def test_alignment_preprocessing_config_defaults(self):
        """method='principal_axes' yields velocity_threshold=0.5 and velocity_metric='mean'."""
        config = AlignmentPreprocessingConfig(method="principal_axes")
        assert config.method == "principal_axes"
        assert config.velocity_threshold == 0.5
        assert config.velocity_metric == "mean"

    def test_alignment_preprocessing_config_rejects_invalid_method(self):
        """An unknown method value is rejected by the Literal annotation."""
        with pytest.raises(ValidationError):
            AlignmentPreprocessingConfig(method="invalid")

    def test_eval_config_alignment_preprocessing_defaults_none(self, tmp_path):
        """EvalConfig.alignment_preprocessing defaults to None (backward compatible)."""
        config = EvalConfig(data_path=str(tmp_path / "x.mat"))
        assert config.alignment_preprocessing is None
