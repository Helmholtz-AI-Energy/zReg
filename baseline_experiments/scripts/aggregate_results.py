"""Aggregate eval_report.json / best_params.json across the whole suite.

Walks baseline_experiments/experiments/<phase>/<name>/ for every run defined
in run_all.py, pulls out the headline metrics (StageMetrics fields) plus the
final params, and writes a single comparison table so the four pipeline
tracks (selfcal, baseline_no_hpo, ground_truth, baseline_with_selfcal) can be
read side by side.

Usage
-----
    python baseline_experiments/scripts/aggregate_results.py
    python baseline_experiments/scripts/aggregate_results.py --output-dir /custom/path
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = SUITE_ROOT / "experiments"

METRIC_FIELDS = [
    "chamfer_distance",
    "hausdorff_distance",
    "path_smoothness",
    "temporal_stability",
    "f1_score",
    "knn_consistency",
]

# Scoped to a single pair (ew06_vs_shah) for baseline_no_hpo/baseline_with_selfcal
# — see run_all.py's BASELINE_NO_HPO/BASELINE_WITH_SELFCAL comment for why.
RUNS = [
    ("selfcal", "kobitski_ew06_alignment"),
    ("selfcal", "shah_alignment"),
    ("selfcal", "shah_label_transfer"),
    ("baseline_no_hpo", "ew06_vs_shah"),
    ("ground_truth", "kobitski_ew06"),
    ("ground_truth", "shah_sample1"),
    ("baseline_with_selfcal", "ew06_vs_shah"),
]


def _load_run(phase: str, name: str) -> dict | None:
    run_dir = EXPERIMENTS_ROOT / phase / name
    report_path = run_dir / "eval_report.json"
    if not report_path.exists():
        return None

    with open(report_path) as f:
        report = json.load(f)

    row = {"phase": phase, "name": name}
    metrics = report.get("metrics", {})
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field)

    score_path = run_dir / "best_params.json"
    row["had_hpo"] = score_path.exists()
    if score_path.exists():
        with open(run_dir / "search_history.json") as f:
            history = json.load(f)
        row["best_score"] = max((t.get("score") for t in history), default=None)

    row["params_json"] = json.dumps(report.get("params", {}), sort_keys=True)
    return row


def build_summary() -> list[dict]:
    rows = []
    for phase, name in RUNS:
        row = _load_run(phase, name)
        if row is not None:
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["phase", "name", *METRIC_FIELDS, "had_hpo", "best_score", "params_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = ["phase", "name", *METRIC_FIELDS, "had_hpo", "best_score"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate baseline_experiments results")
    parser.add_argument("--output-dir", default=str(EXPERIMENTS_ROOT / "summary"))
    args = parser.parse_args(argv)

    rows = build_summary()
    if not rows:
        print("No eval_report.json files found yet under", EXPERIMENTS_ROOT)
        return 0

    out_dir = Path(args.output_dir)
    write_csv(rows, out_dir / "summary.csv")
    write_markdown(rows, out_dir / "summary.md")
    print(f"Aggregated {len(rows)}/{len(RUNS)} completed runs -> {out_dir}/summary.{{csv,md}}")
    missing = [f"{p}/{n}" for p, n in RUNS if not any(r['phase'] == p and r['name'] == n for r in rows)]
    if missing:
        print("Not yet complete:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
