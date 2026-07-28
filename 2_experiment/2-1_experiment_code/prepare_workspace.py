"""
Stage the released data into the working layout the stage scripts expect.

Run this once after cloning. Without it the stage scripts fail immediately with
FileNotFoundError, because they were written against the authors' local working
tree and read from config.DATASET_DIR and config.OUTPUT_DIR, not from the
released folders.
The released files also carry publication names that differ from the pipeline's
internal names. This script bridges both gaps, so the analysis code itself is
published exactly as it was run, with no paths edited after the fact.

What it writes (both destinations are derived from config.py, so they follow
whatever layout config.py declares):

  config.DATASET_DIR/{train,valid,test,test2}.jsonl
      split back out of 1_dataset/raw/dui_cases_1644.jsonl, with the split tag this
      release added removed again, so the files match the source corpus exactly.

  config.OUTPUT_DIR/
      stage2_features_verified.parquet <- 1_dataset/benchmark/benchmark_features.csv
      stage3_oof_predictions.parquet   <- 2_experiment/2-2_experiment_result/predictions/ml_oof_predictions.csv
      stage6_llm_predictions.parquet   <- 2_experiment/2-2_experiment_result/predictions/llm_predictions.csv
      llm_cache_<model>_<variant>.json <- 2_experiment/2-2_experiment_result/llm_caches/ (names unchanged)
      stage3_cv_results.csv, ...       <- 2_experiment/2-2_experiment_result/results/, which publishes each
                                          table under a name that points at the
                                          manuscript; see RESULT_TABLES below

stage1_prepared.parquet is deliberately NOT written here. Once the source splits
are staged, stage1_data_prep.py produces it itself, which is the faithful route.
Run it first, before any other stage.

Usage:
    python 2_experiment/2-1_experiment_code/prepare_workspace.py [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

import config

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "1_dataset"
RESULT = REPO / "2_experiment" / "2-2_experiment_result"

SOURCE_SPLITS = ["train", "valid", "test", "test2"]

# The result tables are published under names that point at the manuscript, while
# the stage scripts read and write the pipeline's own names. This is the mapping.
RESULT_TABLES = {
    "Sec3-1_leakage_audit.csv":                  "disposition_audit.csv",
    "Sec4-1_Table2_ml_performance_per_fold.csv": "stage3_cv_results.csv",
    "Sec4-1_h1_significance_tests.csv":          "stage4_h1_results.csv",
    "Sec4-2_Table3_ml_vs_llm_mcnemar.csv":       "stage7_h2_results.csv",
    "Sec4-2_qwk_difference_bootstrap.csv":       "stage7b_qwk_bootstrap.csv",
    "Sec4-4_Figure3_shap_factor_ranking.csv":    "stage5_factor_ranking.csv",
    "Sec4-4_Table4_explanation_agreement.csv":   "stage8_agreement.csv",
    "Sec4-4_llm_factor_rankings.csv":            "stage8_llm_rankings.csv",
    "Sec4-4_tau_bootstrap_kernelshap.csv":       "stage8b_robustness.csv",
    "Sec4-5_temporal_performance.csv":           "stage9_temporal_results.csv",
    "Sec4-5_temporal_ablation.csv":              "stage9b_ablation.csv",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing released file: {path.relative_to(REPO)}")
    return path


def write_source_splits(dest: Path, force: bool) -> None:
    """Reconstruct the four source-corpus split files from the released subset."""
    dest.mkdir(parents=True, exist_ok=True)
    existing = [s for s in SOURCE_SPLITS if (dest / f"{s}.jsonl").exists()]
    if existing and not force:
        print(f"  skip {dest} (already populated; use --force to overwrite)")
        return

    buckets: dict[str, list[str]] = {s: [] for s in SOURCE_SPLITS}
    with require(DATASET / "raw" / "dui_cases_1644.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            split = rec.pop("split")          # added by this release, not in the source
            buckets[split].append(json.dumps(rec, ensure_ascii=False))

    for split, lines in buckets.items():
        path = dest / f"{split}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"  wrote {path.name}  ({len(lines):,} records)")


def csv_to_parquet(src: Path, out: Path, name: str) -> None:
    df = pd.read_csv(require(src), encoding="utf-8-sig")
    df.to_parquet(out / name, index=False)
    print(f"  wrote {name}  ({len(df):,} rows)")


def copy_tree(src: Path, out: Path, pattern: str) -> None:
    for path in sorted(require(src).glob(pattern)):
        shutil.copy2(path, out / path.name)
        print(f"  copied {path.name}")


def copy_result_tables(src: Path, out: Path) -> None:
    """Copy each published result table under the name the stage scripts expect."""
    for published, pipeline in sorted(RESULT_TABLES.items()):
        shutil.copy2(require(src / published), out / pipeline)
        print(f"  copied {published}  ->  {pipeline}")
    extra = {p.name for p in src.glob("*.csv")} - set(RESULT_TABLES)
    if extra:
        print(f"  note: not mapped, left alone: {', '.join(sorted(extra))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that are already staged")
    args = ap.parse_args()

    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    print(f"source corpus -> {config.DATASET_DIR}")
    write_source_splits(config.DATASET_DIR, args.force)

    print(f"pipeline workspace -> {out}")
    csv_to_parquet(DATASET / "benchmark" / "benchmark_features.csv", out,
                   "stage2_features_verified.parquet")
    csv_to_parquet(RESULT / "predictions" / "ml_oof_predictions.csv", out,
                   "stage3_oof_predictions.parquet")
    csv_to_parquet(RESULT / "predictions" / "llm_predictions.csv", out,
                   "stage6_llm_predictions.parquet")
    copy_tree(RESULT / "llm_caches", out, "llm_cache_*.json")
    copy_result_tables(RESULT / "results", out)

    print("\nWorkspace ready. Run 2_experiment/2-1_experiment_code/stage1_data_prep.py first, then the rest in")
    print("stage order. Re-running stage6_llm.py is optional: the caches staged")
    print("above already reproduce every reported LLM number without API calls.")


if __name__ == "__main__":
    main()
