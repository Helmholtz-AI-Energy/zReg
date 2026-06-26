#!/usr/bin/env python3
"""
Create a comparison PDF for two evaluation runs.

Usage:
    python create_comparison_pdf.py <run1_dir> <run2_dir> <run1_label> <run2_label> [output_name]

Example:
    python create_comparison_pdf.py \
        experiments/runs/selfcal_kobitski/2026-06-22_15-58-35 \
        experiments/runs/selfcal_shah/2026-06-22_15-30-00 \
        "Kobitski" "Shah" \
        comparison_selfcal.pdf
"""

import json
import yaml
import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dicts for display."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_run_data(run_dir):
    """Load config, report, and images from a run directory."""
    run_path = Path(run_dir)

    # Load config
    config_path = run_path / "run_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load report
    report_path = run_path / "eval_report.json"
    with open(report_path) as f:
        report = json.load(f)

    # Load images if they exist
    images = {}
    for img_type in ['alignment_trajectory', 'label_trajectory']:
        img_path = run_path / f"{img_type}.png"
        if img_path.exists():
            images[img_type] = Image.open(img_path)

    return config, report, images


def create_comparison_pdf(run1_dir, run2_dir, label1, label2, output_path):
    """Create a comparison PDF for two runs."""

    # Load data
    config1, report1, images1 = load_run_data(run1_dir)
    config2, report2, images2 = load_run_data(run2_dir)

    # Flatten configs for table
    flat_config1 = flatten_dict(config1)
    flat_config2 = flatten_dict(config2)
    all_keys = sorted(set(flat_config1.keys()) | set(flat_config2.keys()))

    # Create PDF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        # Page 1: Config & Metrics Table
        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(111)
        ax.axis('off')

        fig.suptitle(f'Comparison: {label1} vs {label2}', fontsize=16, fontweight='bold', y=0.98)

        # Build table
        table_data = [['Parameter', label1, label2]]

        # Add config params
        for key in all_keys:
            val1 = flat_config1.get(key, 'N/A')
            val2 = flat_config2.get(key, 'N/A')

            # Format values
            if isinstance(val1, (list, dict)):
                val1 = str(val1)
            if isinstance(val2, (list, dict)):
                val2 = str(val2)

            val1_str = str(val1)
            val2_str = str(val2)

            # Truncate long values
            if len(val1_str) > 45:
                val1_str = val1_str[:42] + '...'
            if len(val2_str) > 45:
                val2_str = val2_str[:42] + '...'

            table_data.append([key, val1_str, val2_str])

        # Add metrics section
        table_data.append(['', '', ''])
        table_data.append(['METRICS', '', ''])

        metrics_keys = [
            'chamfer_distance', 'hausdorff_distance', 'path_smoothness',
            'temporal_stability', 'f1_score', 'knn_consistency'
        ]

        for key in metrics_keys:
            val1 = report1.get('metrics', {}).get(key, 'N/A')
            val2 = report2.get('metrics', {}).get(key, 'N/A')

            if isinstance(val1, float):
                val1 = f"{val1:.6g}"
            if isinstance(val2, float):
                val2 = f"{val2:.6g}"

            table_data.append([key, str(val1), str(val2)])

        # Create table
        table = ax.table(
            cellText=table_data,
            cellLoc='left',
            loc='center',
            colWidths=[0.35, 0.32, 0.32]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style metrics header
        metrics_row = len(table_data) - len(metrics_keys) - 1
        for i in range(3):
            table[(metrics_row, i)].set_facecolor('#D9E1F2')
            table[(metrics_row, i)].set_text_props(weight='bold')

        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#F2F2F2' if i % 2 == 0 else 'white'
            for j in range(3):
                table[(i, j)].set_facecolor(color)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2: Alignment Trajectories (if available)
        if 'alignment_trajectory' in images1 and 'alignment_trajectory' in images2:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Alignment Trajectory Comparison', fontsize=14, fontweight='bold')

            axes[0].imshow(images1['alignment_trajectory'])
            axes[0].set_title(label1, fontsize=12, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(images2['alignment_trajectory'])
            axes[1].set_title(label2, fontsize=12, fontweight='bold')
            axes[1].axis('off')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Page 3: Label Trajectories (if available)
        if 'label_trajectory' in images1 and 'label_trajectory' in images2:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Label Transfer Trajectory Comparison', fontsize=14, fontweight='bold')

            axes[0].imshow(images1['label_trajectory'])
            axes[0].set_title(label1, fontsize=12, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(images2['label_trajectory'])
            axes[1].set_title(label2, fontsize=12, fontweight='bold')
            axes[1].axis('off')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"✓ PDF created: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    run1_dir = sys.argv[1]
    run2_dir = sys.argv[2]
    label1 = sys.argv[3]
    label2 = sys.argv[4]
    output = sys.argv[5] if len(sys.argv) > 5 else "comparison.pdf"

    create_comparison_pdf(run1_dir, run2_dir, label1, label2, output)
