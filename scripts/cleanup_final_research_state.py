from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


FINAL_DOCS = {
    "01_dataset_and_protocol.md": """# Dataset And Protocol

## Objective

DS-Fall-RD studies direction-sensitive fall detection from wrist IMU signals.
The final benchmark is restricted to BITS and WEDA.

## Benchmark

- Datasets: BITS, WEDA.
- Sampling rate: 25 Hz.
- Window: 2 seconds, event-centered.
- Temporal input: 50 x 12 tilt12.
- Direction classes: forward, backward, lateral.
- Direction metrics are computed only on supervised fall windows.

## E1-E7 Protocol

| ID | Train | Validation | Test |
| --- | --- | --- | --- |
| E1_BITS_TO_BITS | BITS train | BITS val | BITS test |
| E2_WEDA_TO_WEDA | WEDA train | WEDA val | WEDA test |
| E3_BITS_WEDA_MIXED | BITS+WEDA train | BITS+WEDA val | BITS+WEDA test |
| E4_BITS_TO_WEDA | BITS train | BITS val | WEDA test |
| E5_WEDA_TO_BITS | WEDA train | WEDA val | BITS test |
| E6_E3_MIXED_TEST_BITS | reuse E3 | reuse E3 | BITS test |
| E7_E3_MIXED_TEST_WEDA | reuse E3 | reuse E3 | WEDA test |

FullTiming features are treated only as an upper bound. They are not used as
the main result because they can encode event-centered timing cues.
""",
    "02_processed_dataset.md": """# Processed Dataset

Processed data is expected under `data/processed/` and final feature summaries
are loaded from `outputs/reports/ml_feature_analysis_25hz/summary_feature_matrix.csv`.

## Labels

- `fall_label`: binary non-fall/fall target.
- `direction_label`: forward, backward, lateral, or none.
- `direction_supervised`: whether direction loss/metric is valid.
- `dataset`: bits or weda for the final benchmark.
- `split`: train, val, or test.
- `subject_id`: used when available for audit and reporting.

## Temporal Feature Set

The main temporal input is tilt12:

- ax, ay, az
- gx, gy, gz
- acc_mag
- gyro_mag
- jerk
- roll
- pitch
- tilt_delta

## Summary Feature Sets

- FallNoTiming_Core: compact fall summary features.
- FallNoTiming_Full: all signal-derived summary features without timing indexes.
- DirectionCore: signed gyro/acc/roll/pitch direction features.
- DirectionFullNoTiming: all signal-derived direction features without timing indexes.
- FullTiming: upper-bound only, not a main result.
""",
    "03_model_architecture_a5.md": """# A5 Model Architecture

A5WCEFW is the DS-Fall-RD deep temporal multitask reference.

## Input

- Shape: 50 x 12.
- Feature set: tilt12.

## Structure

- Dual accelerometer/gyroscope stream encoder.
- Depthwise-separable temporal convolution blocks.
- Shared temporal encoder.
- Task-specific attention pooling.
- Fall head: non-fall/fall.
- Direction head: forward/backward/lateral.
- Weighted cross entropy for fall and direction in the reference setup.

## Current Role

A5 remains the main deep temporal baseline and the safest direction expert.
Its fall head is retained as an auxiliary/reference result when the final
hybrid pipeline uses a separate ML fall expert.
""",
    "04_hybrid_architecture.md": """# Hybrid Architecture

Fall detection and direction classification respond to different evidence.

- Fall detection depends strongly on impact magnitude, jerk, post-impact
  stability, and compact summary statistics.
- Direction classification depends on signed temporal motion, roll/pitch
  transitions, and gyro-axis patterns.

The final practical pipeline separates the two experts:

- ML Fall Expert: trained on no-timing summary features.
- A5 Direction Expert: keeps the deep temporal direction head.

FullTiming is not used in the main hybrid model. It remains an upper-bound
analysis only.
""",
    "05_final_four_pipelines.md": """# Final Four Pipelines

| Pipeline | Fall decision | Direction decision | Feature/Input | Role |
| --- | --- | --- | --- | --- |
| A5_reference | A5 fall head | A5 direction head | tilt12 sequence | deep multitask reference |
| Hybrid_NoTiming | GradientBoosting on FallNoTiming_Full | A5 direction head | summary + tilt12 | main practical architecture |
| EdgeLite_Hybrid | RandomForest on FallNoTiming_Core | A5 direction head | compact summary + tilt12 | compact ablation |
| ML_only_separated | GradientBoosting on FallNoTiming_Full | GradientBoosting on DirectionFullNoTiming | summary only | analysis pipeline |

ML_only_separated is not the main paper architecture unless its direction
expert is consistently better than A5 across E1-E7.
""",
    "06_research_summary_from_start_to_now.md": """# Research Summary From Start To Now

DS-Fall started as a lightweight dual-stream IMU fall detector. DS-Fall-RD
added rotation-aware features, task-specific attention, and weighted losses to
improve direction-sensitive recognition.

A5WCEFW became the strongest deep reference. It is especially strong for
direction, but its fall head creates many WEDA false positives. Data analysis
showed that WEDA contains harder ADL negatives, so the WEDA fall gap is not
only a model-size issue.

Feature analysis showed that no-timing classical fall experts can reduce WEDA
false positives. Direction remains safer with A5 because signed temporal
patterns are more consistently handled by the deep direction head.

The final decision is a task-separated hybrid: no-timing ML fall expert plus
A5 direction expert. FullTiming remains an upper-bound analysis.
""",
    "07_final_e1_e7_results_summary.md": """# Final E1-E7 Results Summary

This file is overwritten by `scripts/run_final_four_pipelines_e1_e7.py` after
the final E1-E7 run.
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final DS-Fall-RD research folders and clean documentation.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--archive-logs",
        action="store_true",
        help="Move old outputs/logs/*.log files into archive/old_experiments/logs_TIMESTAMP.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_docs(final_dir: Path) -> None:
    ensure_dir(final_dir)
    for name, content in FINAL_DOCS.items():
        (final_dir / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def archive_logs(repo_root: Path, archive_root: Path) -> list[str]:
    log_dir = repo_root / "outputs" / "logs"
    moved: list[str] = []
    if not log_dir.exists():
        return moved
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_dir = ensure_dir(archive_root / f"logs_{stamp}")
    for path in log_dir.glob("*.log"):
        dst = dst_dir / path.name
        shutil.move(str(path), str(dst))
        moved.append(str(dst.relative_to(repo_root)))
    return moved


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    final_output = repo_root / "outputs" / "final_e1_e7"
    archive_root = repo_root / "archive"

    dirs = [
        repo_root / "docs" / "final",
        final_output,
        final_output / "reports",
        final_output / "tables",
        final_output / "figures",
        final_output / "figures" / "confusion_matrices",
        final_output / "figures" / "bar_charts",
        final_output / "predictions",
        archive_root / "old_experiments",
        archive_root / "old_reports",
    ]
    for path in dirs:
        ensure_dir(path)

    write_docs(repo_root / "docs" / "final")
    moved_logs = archive_logs(repo_root, archive_root / "old_experiments") if args.archive_logs else []

    manifest = {
        "repo_root": str(repo_root),
        "created_or_verified": [str(path.relative_to(repo_root)) for path in dirs],
        "moved_logs": moved_logs,
        "hard_delete": False,
        "note": "Raw data, processed data, source code, scripts, notebooks, model artifacts, and reports were not deleted.",
    }
    manifest_path = final_output / "reports" / "cleanup_final_research_state_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Final research folders ready.")
    print(f"Docs: {repo_root / 'docs' / 'final'}")
    print(f"Manifest: {manifest_path}")
    if moved_logs:
        print(f"Archived {len(moved_logs)} log files.")
    else:
        print("No files were archived. Use --archive-logs to move old log files only.")


if __name__ == "__main__":
    main()
