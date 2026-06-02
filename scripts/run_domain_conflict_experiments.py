from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import make_config
from src.experiments.domain_conflict import DEFAULT_DATASETS, run_domain_conflict_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run domain-conflict experiments with fixed A5WCEFW.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--processed-x-is-raw", action="store_true", help="Use if data/processed/X.npy is raw6, not normalized.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, help="Datasets to include.")
    parser.add_argument(
        "--suite",
        nargs="*",
        default=["dataset_specific", "train3_test1"],
        choices=["dataset_specific", "train3_test1", "train1_testothers"],
        help="Experiment suites to run.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.project_root)
    run_domain_conflict_suite(
        config,
        datasets=args.datasets,
        suites=args.suite,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        processed_x_is_normalized=not args.processed_x_is_raw,
    )


if __name__ == "__main__":
    main()
