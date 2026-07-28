"""
Build the human- and machine-readable data release (CSV / JSON / JSONL).

The journal requires the data underlying the results to be supplied in an open,
readable format (CSV, JSON, TXT, ...), so every Parquet table shipped in this
repository is mirrored as CSV/JSON here, and the drunk-driving subset of the
source corpus is written out as JSONL and CSV.

Inputs
------
  <source>/{train,valid,test,test2}.jsonl   the LBox Open DUI subset (1,644 records)
  1_dataset/benchmark/benchmark_features.parquet the attorney-verified feature table
  1_dataset/benchmark/splits.csv                 id / split / group_key for the 1,500 unique cases
  2_experiment/2-2_experiment_result/predictions/*.parquet                ML out-of-fold and LLM predictions

Outputs (all UTF-8; CSV written with a BOM so spreadsheet software reads Korean text)
-------
  1_dataset/raw/dui_cases_1644.jsonl        all 1,644 records, original nested structure
  1_dataset/raw/dui_cases_1644.csv          same records, nested fields flattened
  1_dataset/raw/dui_cases_1500_unique.jsonl 1,500 unique cases (test2 duplicates removed)
  1_dataset/raw/dui_cases_1500_unique.csv
  1_dataset/benchmark/benchmark_features.{csv,json}
  2_experiment/2-2_experiment_result/predictions/ml_oof_predictions.csv
  2_experiment/2-2_experiment_result/predictions/llm_predictions.{csv,jsonl}
  1_dataset/codebook.md                     column definitions for every released table

Usage
-----
  python 2_experiment/2-1_experiment_code/make_data_release_files.py [--source PATH_TO_dataset_FOLDER]

Note: --source points at the local copy of the DUI subset built by
lbox_dui_dataset_builder.py. It is not part of this repository; the step is
recorded here so the release files are reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO.parent.parent / "dataset"

SPLITS = ["train", "valid", "test", "test2"]

# Nested source fields -> flat CSV columns.
FLAT_MAP = {
    "label_text": ("label", "text"),
    "fine_lv": ("label", "fine_lv"),
    "imprisonment_with_labor_lv": ("label", "imprisonment_with_labor_lv"),
    "imprisonment_without_labor_lv": ("label", "imprisonment_without_labor_lv"),
    "ruling_text": ("ruling", "text"),
}
RULING_PARSE_MAP = {
    "ruling_fine_type": ("fine", "type"),
    "ruling_fine_unit": ("fine", "unit"),
    "ruling_fine_value": ("fine", "value"),
    "ruling_imprisonment_type": ("imprisonment", "type"),
    "ruling_imprisonment_unit": ("imprisonment", "unit"),
    "ruling_imprisonment_value": ("imprisonment", "value"),
}

CLASS_NAMES = {0: "fine/none", 1: "short", 2: "medium", 3: "long"}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV with a UTF-8 BOM: Excel misreads Korean text without it."""
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"  wrote {path.relative_to(REPO)}  ({len(df):,} rows x {df.shape[1]} cols)")


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(REPO)}  ({len(records):,} records)")


def write_json(df: pd.DataFrame, path: Path) -> None:
    records = json.loads(df.to_json(orient="records"))
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"  wrote {path.relative_to(REPO)}  ({len(records):,} records)")


def load_source(source: Path) -> list[dict]:
    """Read the four split files, tagging each record with its split name."""
    records = []
    for split in SPLITS:
        path = source / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"source file missing: {path}")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                records.append({"id": rec["id"], "split": split, **{k: v for k, v in rec.items() if k != "id"}})
    return records


def flatten(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {k: rec[k] for k in ("id", "split", "casetype", "casename")}
        row["facts"] = rec.get("facts", "")
        row["reason"] = rec.get("reason", "")
        for col, (outer, inner) in FLAT_MAP.items():
            row[col] = rec.get(outer, {}).get(inner)
        parse = rec.get("ruling", {}).get("parse", {})
        for col, (outer, inner) in RULING_PARSE_MAP.items():
            row[col] = parse.get(outer, {}).get(inner)
        # Extra columns carried on the de-duplicated file only.
        for extra in ("class4", "class4_name", "group_key"):
            if extra in rec:
                row[extra] = rec[extra]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="folder holding {train,valid,test,test2}.jsonl of the DUI subset")
    args = ap.parse_args()

    raw_dir = REPO / "1_dataset" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"source: {args.source}")
    records = load_source(args.source)
    print(f"loaded {len(records):,} records")

    lvs = sorted({r["label"]["imprisonment_with_labor_lv"] for r in records})
    print(f"  imprisonment_with_labor_lv values present: {lvs}")

    # ---- 1. full 1,644-record subset -------------------------------------
    write_jsonl(records, raw_dir / "dui_cases_1644.jsonl")
    write_csv(flatten(records), raw_dir / "dui_cases_1644.csv")

    # ---- 2. de-duplicated 1,500-case analysis set -------------------------
    # test2 is a strict subset of test (same id, same facts, same label), so the
    # 1,500 unique cases are train + valid + test. class4 and group_key are
    # attached from the released benchmark tables.
    splits = pd.read_csv(REPO / "1_dataset" / "benchmark" / "splits.csv")
    bench = pd.read_parquet(REPO / "1_dataset" / "benchmark" / "benchmark_features.parquet")
    group_key = dict(zip(splits["id"], splits["group_key"]))
    class4 = dict(zip(bench["id"], bench["class4"]))

    unique = []
    for rec in records:
        if rec["split"] == "test2":
            continue
        cls = int(class4[rec["id"]])
        enriched = {**rec, "class4": cls, "class4_name": CLASS_NAMES[cls],
                    "group_key": int(group_key[rec["id"]])}
        unique.append(enriched)

    assert len(unique) == 1500, f"expected 1,500 unique cases, got {len(unique)}"
    assert len({r["id"] for r in unique}) == 1500, "duplicate ids in the unique set"
    assert {r["id"] for r in unique} == set(splits["id"]), "unique ids != splits.csv ids"

    write_jsonl(unique, raw_dir / "dui_cases_1500_unique.jsonl")
    write_csv(flatten(unique), raw_dir / "dui_cases_1500_unique.csv")

    # ---- 3. Parquet tables mirrored as CSV / JSON -------------------------
    bench_out = REPO / "1_dataset" / "benchmark"
    write_csv(bench, bench_out / "benchmark_features.csv")
    write_json(bench, bench_out / "benchmark_features.json")

    pred_dir = REPO / "2_experiment" / "2-2_experiment_result" / "predictions"
    ml = pd.read_parquet(pred_dir / "ml_oof_predictions.parquet")
    write_csv(ml, pred_dir / "ml_oof_predictions.csv")

    llm = pd.read_parquet(pred_dir / "llm_predictions.parquet")
    write_csv(llm, pred_dir / "llm_predictions.csv")
    write_jsonl(json.loads(llm.to_json(orient="records", force_ascii=False)),
                pred_dir / "llm_predictions.jsonl")

    print("done.")


if __name__ == "__main__":
    main()
