from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import make_config
from src.reporting.analysis_notes import update_analysis_notes_from_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update analysis_notes.md from cross-dataset experiment reports.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--direction-threshold", type=float, default=0.60)
    parser.add_argument("--fall-threshold", type=float, default=0.70)
    parser.add_argument("--reports", nargs="*", default=None, help="Optional explicit CSV report paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.project_root)
    path = update_analysis_notes_from_reports(
        project_root=config.project_root,
        output_dir=config.output_dir,
        report_paths=args.reports,
        direction_threshold=args.direction_threshold,
        fall_threshold=args.fall_threshold,
    )
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
