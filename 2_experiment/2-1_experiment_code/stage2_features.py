"""
Stage 2 - Apply the frozen feature extractor, report coverage, emit a manual-validation sample.

Inputs : outputs/stage1_prepared.parquet
Outputs:
  - stage2_features.parquet         : id, split, class4, group_key, 5 ML-A features
  - stage2_coverage.md              : extraction coverage + feature distributions
  - stage2_validation_sample.csv    : random 100 cases (facts excerpt + extracted values)
                                      for manual checking by a domain expert (lawyer).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

import config
import features as F

config.set_seed()

df = pd.read_parquet(config.OUTPUT_DIR / "stage1_prepared.parquet")
print(f"Loaded {len(df)} cases from stage1_prepared.parquet")

# -- Apply the frozen extractor -------------------------------------------
feat = df["facts"].apply(F.extract_features).apply(pd.Series)
out = pd.concat([df[["id", "split", "class4", "group_key", "facts"]], feat], axis=1)

# Coverage = fraction with a successfully extracted (non-missing) value.
def coverage(col, missing_pred):
    miss = out[col].apply(missing_pred).sum()
    return (len(out) - miss) / len(out)

cov = {
    "bac": coverage("bac", lambda v: pd.isna(v)),
    "distance_km": coverage("distance_km", lambda v: pd.isna(v)),
    "vehicle": coverage("vehicle", lambda v: v == "other"),          # "other" = unmatched
    "prior_dui_count": coverage("prior_dui_count", lambda v: False), # always defined (>=0)
    "offense_year": coverage("offense_year", lambda v: pd.isna(v) or v is None),
}

print("\n=== Extraction coverage ===")
for k, v in cov.items():
    print(f"  {k:16s} {v*100:5.1f}%")

# -- Distributions ---------------------------------------------------------
print("\n=== Feature summaries ===")
print(f"  bac:           mean={out['bac'].mean():.4f}, "
      f"min={out['bac'].min():.3f}, max={out['bac'].max():.3f}, n_missing={out['bac'].isna().sum()}")
print(f"  distance_km:   median={out['distance_km'].median():.2f}, "
      f"max={out['distance_km'].max():.1f}, n_missing={out['distance_km'].isna().sum()}")
print(f"  vehicle:       {dict(out['vehicle'].value_counts())}")
print(f"  prior_dui_cnt: {dict(out['prior_dui_count'].value_counts().sort_index())}")
print(f"  offense_year:  n_missing={out['offense_year'].isna().sum()}")

# -- Save feature table (drop facts text from the modeling frame) ----------
keep = ["id", "split", "class4", "group_key"] + config.FEATURES_A
out_parquet = config.OUTPUT_DIR / "stage2_features.parquet"
out[keep].to_parquet(out_parquet, index=False)
print(f"\nSaved: {out_parquet}")

# -- Manual-validation sample (random 100, deduplicated by group) ----------
uniq = out.drop_duplicates(subset="group_key")
sample = uniq.sample(n=min(150, len(uniq)), random_state=config.SEED).copy()
# full facts text (whitespace-normalized) so the reviewer is never cut off mid-sentence
sample["facts_full"] = sample["facts"].str.replace(r"\s+", " ", regex=True)
val_cols = ["id", "split", "class4"] + config.FEATURES_A + ["facts_full"]
val_csv = config.OUTPUT_DIR / "stage2_validation_sample.csv"
sample[val_cols].to_csv(val_csv, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel
print(f"Saved: {val_csv}  ({len(sample)} cases for manual review)")

# -- Coverage report (markdown) -------------------------------------------
lines = ["# Stage 2 - Feature extraction coverage & distributions\n"]
lines.append(f"Cases: {len(out)} (modeling frame keeps the 5 ML-A features).\n")
lines.append("\n## Coverage (successful extraction)\n")
lines.append("| feature | coverage |\n|---|---|")
for k, v in cov.items():
    lines.append(f"| {k} ({config.FEATURES_A_EN[k]}) | {v*100:.1f}% |")
lines.append("\n## Distributions\n")
lines.append(f"- **bac**: mean {out['bac'].mean():.4f}, range "
             f"[{out['bac'].min():.3f}, {out['bac'].max():.3f}], missing {out['bac'].isna().sum()}")
lines.append(f"- **distance_km**: median {out['distance_km'].median():.2f}, "
             f"max {out['distance_km'].max():.1f}, missing {out['distance_km'].isna().sum()}")
lines.append(f"- **vehicle**: {dict(out['vehicle'].value_counts())}")
lines.append(f"- **prior_dui_count**: {dict(out['prior_dui_count'].value_counts().sort_index())}")
lines.append(f"- **offense_year**: missing {out['offense_year'].isna().sum()}")
lines.append("\n## Manual validation\n")
lines.append(f"A random sample of {len(sample)} unique cases is exported to "
             f"`stage2_validation_sample.csv` for manual checking by a domain expert. "
             f"Per-feature precision from that review is reported in the paper "
             f"(plan v4, Sections 2 & 9). NOTE: vehicle 'other' and missing bac/distance are "
             f"counted as extraction gaps, not errors, and are imputed in Stage 3.")
(config.OUTPUT_DIR / "stage2_coverage.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {config.OUTPUT_DIR / 'stage2_coverage.md'}")
print("\nStage 2 done.")
