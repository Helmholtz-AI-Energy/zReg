#!/usr/bin/env python3
"""Render first / middle / last frame triptych for every synthetic dataset.

Output: reports/dataset_previews/{fully_synthetic,semi_synthetic}/<name>.png
Each PNG: 1 row × 3 3-D scatter plots (first, mid, last frame).
"""

from pathlib import Path

from eval.viz import render_dataset_triptych

ROOT      = Path(__file__).resolve().parent.parent
FULLY_DIR = ROOT / "data" / "synthetic" / "fully_synthetic"
SEMI_DIR  = ROOT / "data" / "synthetic" / "semi_synthetic"
OUT_BASE  = ROOT / "reports" / "dataset_previews"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_datasets(base_dir: Path) -> list[tuple[str, Path]]:
    """Return (name, csv_path) pairs from all immediate subdirs of base_dir."""
    result = []
    for sub in sorted(base_dir.iterdir()):
        csv = sub / f"{sub.name}.csv"
        if csv.exists():
            result.append((sub.name, csv))
    return result


def main() -> None:
    datasets = [
        ("fully_synthetic", collect_datasets(FULLY_DIR)),
        ("semi_synthetic",  collect_datasets(SEMI_DIR)),
    ]

    total = sum(len(ds) for _, ds in datasets)
    done  = 0

    for category, items in datasets:
        out_dir = OUT_BASE / category
        print(f"\n{'='*60}")
        print(f"  {category}  ({len(items)} datasets)")
        print(f"{'='*60}")
        for name, csv_path in items:
            done += 1
            print(f"[{done}/{total}]  {name}")
            p = render_dataset_triptych(csv_path, name, out_dir)
            print(f"  saved  {p.relative_to(ROOT)}")

    print(f"\nDone — {done} PNGs written to {OUT_BASE.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
