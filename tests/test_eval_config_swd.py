"""Tests for EvalConfig SWD alignment configuration validation.

Tests for alignment_method='swd' and swd_variant field validation.
Verifies that alignment_method and swd_variant work correctly together,
and that alignment_method is independent of dtw_dist_fn (Pitfall 1).
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from eval.config import EvalConfig, EvalConfigError


class TestEvalConfigSWDBasics:
    """Basic SWD configuration acceptance and defaults."""

    def test_eval_config_swd_alignment_method_accepted(self, tmp_path):
        """Test that alignment_method='swd' is accepted."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd"
        )
        assert config.alignment_method == "swd"

    def test_eval_config_swd_variant_field_exists(self, tmp_path):
        """Test that swd_variant field can be set explicitly."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd",
            swd_variant="oswd"
        )
        assert config.swd_variant == "oswd"
        assert config.alignment_method == "swd"

    def test_eval_config_swd_variant_default(self, tmp_path):
        """Test that swd_variant defaults to 'aswd' when alignment_method='swd'."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd"
        )
        assert config.swd_variant == "aswd"
        assert config.alignment_method == "swd"

    def test_eval_config_swd_variant_default_cpd_unchanged(self, tmp_path):
        """Test that swd_variant still defaults to 'aswd' even with alignment_method='cpd'."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="cpd"
        )
        assert config.swd_variant == "aswd"  # Default value
        assert config.alignment_method == "cpd"

    # =====================
    # SWD Variant Validation
    # =====================

    def test_eval_config_swd_variant_invalid_with_swd_method(self, tmp_path):
        """Test that invalid swd_variant raises ValueError when alignment_method='swd'."""
        with pytest.raises(ValueError, match="swd_variant must be one of"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="swd",
                swd_variant="invalid"
            )

    @pytest.mark.parametrize("variant", ["swd", "aswd", "oswd", "gswd", "pswd", "maxswd"])
    def test_eval_config_swd_all_variants_accepted(self, tmp_path, variant):
        """Test that all 6 SWD variants are accepted (parametrized, ALIGN-05-03)."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd",
            swd_variant=variant
        )
        assert config.swd_variant == variant
        assert config.alignment_method == "swd"

    def test_eval_config_swd_variant_empty_string_raises(self, tmp_path):
        """Test that empty string for swd_variant raises ValueError."""
        with pytest.raises(ValueError, match="swd_variant must be one of"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="swd",
                swd_variant=""
            )

    def test_eval_config_swd_variant_case_sensitive(self, tmp_path):
        """Test that swd_variant validation is case-sensitive."""
        with pytest.raises(ValueError, match="swd_variant must be one of"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="swd",
                swd_variant="ASWD"  # uppercase should fail
            )

    # =====================
    # Alignment Method Validation
    # =====================

    def test_eval_config_alignment_method_invalid_raises(self, tmp_path):
        """Test that invalid alignment_method raises ValueError."""
        with pytest.raises(ValueError, match="alignment_method must be"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="invalid_method"
            )

    @pytest.mark.parametrize("method", ["cpd", "icp", "swd"])
    def test_eval_config_alignment_method_all_valid(self, tmp_path, method):
        """Test that all valid alignment methods are accepted (parametrized)."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method=method
        )
        assert config.alignment_method == method

    # =====================
    # Pitfall 1: Independence Tests
    # =====================

    def test_eval_config_swd_variant_ignored_for_cpd(self, tmp_path):
        """Test that swd_variant is not validated when alignment_method='cpd' (Pitfall 1)."""
        # swd_variant can have any value when alignment_method != 'swd'
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="cpd",
            swd_variant="oswd"
        )
        assert config.swd_variant == "oswd"
        assert config.alignment_method == "cpd"

    def test_eval_config_swd_variant_ignored_for_icp(self, tmp_path):
        """Test that swd_variant is not validated when alignment_method='icp' (Pitfall 1)."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="icp",
            swd_variant="gswd"
        )
        assert config.swd_variant == "gswd"
        assert config.alignment_method == "icp"

    def test_eval_config_multiple_alignment_configs(self, tmp_path):
        """Test that multiple alignment methods can be configured."""
        # CPD alignment
        config1 = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="cpd"
        )
        assert config1.alignment_method == "cpd"

        # SWD alignment with specific variant
        config2 = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd",
            swd_variant="oswd"
        )
        assert config2.alignment_method == "swd"
        assert config2.swd_variant == "oswd"

    # =====================
    # Integration Tests
    # =====================

    def test_eval_config_swd_with_other_params(self, tmp_path):
        """Test SWD configuration with various other parameters."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd",
            swd_variant="aswd",
            run_alignment=True,
            val_split=0.3
        )
        assert config.alignment_method == "swd"
        assert config.swd_variant == "aswd"
        assert config.run_alignment is True
        assert config.val_split == 0.3

    def test_eval_config_swd_with_optional_params(self, tmp_path):
        """Test SWD configuration with other optional EvalConfig parameters."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd",
            swd_variant="gswd",
            n_synthetic=200,
            pipeline_mode="paired",
            target_data_path=str(tmp_path / "target.mat")
        )
        assert config.alignment_method == "swd"
        assert config.swd_variant == "gswd"
        assert config.n_synthetic == 200
        assert config.pipeline_mode == "paired"

    def test_eval_config_swd_with_all_parameters(self, tmp_path):
        """Test SWD configuration with comprehensive parameter set."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd",
            swd_variant="pswd",
            n_synthetic=150,
            transform_degree=0.15,
            run_alignment=True,
            run_label_transfer=True,
            search_strategy="bayesian",
            tier="dev",
            n_trials=20,
            output_dir="experiments/swd_runs",
            save_plots=True,
            verbose=True
        )
        assert config.alignment_method == "swd"
        assert config.swd_variant == "pswd"
        assert config.n_synthetic == 150
        assert config.search_strategy == "bayesian"
        assert config.tier == "dev"


class TestEvalConfigSWDYAMLLoading:
    """Tests for loading SWD configuration from YAML files."""

    def test_load_yaml_with_alignment_method_swd(self, tmp_path):
        """Test loading EvalConfig from YAML with alignment_method='swd'."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
            "alignment_method": "swd",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = EvalConfig.from_yaml(str(yaml_path))
        assert config.alignment_method == "swd"
        assert config.swd_variant == "aswd"  # default

    def test_load_yaml_with_swd_variant(self, tmp_path):
        """Test loading YAML with explicit swd_variant."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
            "alignment_method": "swd",
            "swd_variant": "oswd",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = EvalConfig.from_yaml(str(yaml_path))
        assert config.alignment_method == "swd"
        assert config.swd_variant == "oswd"

    def test_load_yaml_with_all_variants(self, tmp_path):
        """Test loading YAML with each SWD variant."""
        variants = ["swd", "aswd", "oswd", "gswd", "pswd", "maxswd"]
        for variant in variants:
            yaml_path = tmp_path / f"config_{variant}.yaml"
            yaml_content = {
                "data_path": str(tmp_path / "data.mat"),
                "alignment_method": "swd",
                "swd_variant": variant,
            }
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_content, f)

            config = EvalConfig.from_yaml(str(yaml_path))
            assert config.swd_variant == variant

    def test_load_yaml_with_invalid_swd_variant_raises(self, tmp_path):
        """Test that YAML with invalid swd_variant raises EvalConfigError."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
            "alignment_method": "swd",
            "swd_variant": "invalid_variant",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        with pytest.raises(EvalConfigError, match="swd_variant"):
            EvalConfig.from_yaml(str(yaml_path))

    def test_load_yaml_swd_without_variant_defaults(self, tmp_path):
        """Test that YAML with alignment_method='swd' but no swd_variant defaults correctly."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
            "alignment_method": "swd",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = EvalConfig.from_yaml(str(yaml_path))
        assert config.alignment_method == "swd"
        assert config.swd_variant == "aswd"

    def test_load_yaml_with_swd_and_other_fields(self, tmp_path):
        """Test loading YAML with alignment_method='swd' and other fields."""
        yaml_path = tmp_path / "config.yaml"
        yaml_content = {
            "data_path": str(tmp_path / "data.mat"),
            "alignment_method": "swd",
            "swd_variant": "gswd",
            "n_synthetic": 200,
            "run_alignment": True,
            "pipeline_mode": "paired",
            "target_data_path": str(tmp_path / "target.mat"),
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = EvalConfig.from_yaml(str(yaml_path))
        assert config.alignment_method == "swd"
        assert config.swd_variant == "gswd"
        assert config.n_synthetic == 200
        assert config.pipeline_mode == "paired"


class TestEvalConfigSWDValidator:
    """Tests for validation logic specific to SWD configuration."""

    def test_swd_variant_validator_with_swd_method(self, tmp_path):
        """Test swd_variant validator triggers only for alignment_method='swd'."""
        # Valid: alignment_method != 'swd', any swd_variant
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="cpd",
            swd_variant="invalid_value"
        )
        # No error raised
        assert config.alignment_method == "cpd"

    def test_swd_variant_validator_rejects_with_swd_method(self, tmp_path):
        """Test swd_variant validator rejects invalid values with alignment_method='swd'."""
        with pytest.raises(ValueError):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="swd",
                swd_variant="not_a_variant"
            )

    def test_alignment_method_validator_rejects_invalid(self, tmp_path):
        """Test alignment_method validator rejects invalid values."""
        with pytest.raises(ValueError, match="alignment_method must be"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="dtw"
            )

    def test_swd_accepts_all_variant_strings(self, tmp_path):
        """Test that all valid variant strings are recognized."""
        valid_variants = ["swd", "aswd", "oswd", "gswd", "pswd", "maxswd"]
        for variant in valid_variants:
            config = EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="swd",
                swd_variant=variant
            )
            assert config.swd_variant == variant


class TestEvalConfigSWDEdgeCases:
    """Edge case and boundary tests for SWD configuration."""

    def test_eval_config_swd_multiple_instances_independent(self, tmp_path):
        """Test that multiple EvalConfig instances don't share state."""
        config1 = EvalConfig(
            data_path=str(tmp_path / "data1.mat"),
            alignment_method="swd",
            swd_variant="aswd"
        )
        config2 = EvalConfig(
            data_path=str(tmp_path / "data2.mat"),
            alignment_method="swd",
            swd_variant="oswd"
        )
        assert config1.swd_variant == "aswd"
        assert config2.swd_variant == "oswd"

    def test_eval_config_swd_preserves_other_defaults(self, tmp_path):
        """Test that setting SWD doesn't affect other defaults."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd",
            swd_variant="pswd"
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

    def test_eval_config_cpd_ignores_swd_variant_value(self, tmp_path):
        """Test that cpd ignores swd_variant even if set."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="cpd",
            swd_variant="aswd"  # This is just stored, not used
        )
        assert config.alignment_method == "cpd"
        # Field is stored but not validated
        assert config.swd_variant == "aswd"

    def test_eval_config_icp_ignores_swd_variant_value(self, tmp_path):
        """Test that ICP ignores swd_variant even if set."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="icp",
            swd_variant="maxswd"  # This is just stored, not used
        )
        assert config.alignment_method == "icp"
        assert config.swd_variant == "maxswd"

    def test_eval_config_switching_alignment_methods(self, tmp_path):
        """Test conceptually switching alignment methods."""
        # First with SWD
        swd_config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="swd",
            swd_variant="oswd"
        )
        assert swd_config.alignment_method == "swd"

        # Then with CPD
        cpd_config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            alignment_method="cpd",
            swd_variant="oswd"  # Can be ignored
        )
        assert cpd_config.alignment_method == "cpd"
        assert not (swd_config.alignment_method == cpd_config.alignment_method)
