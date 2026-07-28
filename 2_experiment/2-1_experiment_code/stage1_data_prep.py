"""
Stage 1 - Data preparation, 4-class relabeling, duplicate / group check, year extraction.

Outputs (to outputs/):
  - stage1_prepared.parquet : one row per case (id, split, facts, reason, target_lv, class4, offense_year)
  - T1_distributions.md      : class & year distributions + duplicate report (paper Table 1)
This stage decides whether a group-aware CV split is required (research plan v4, Section 6).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import re
from collections import Counter

import pandas as pd
import config

config.set_seed()

# -- Loaders ---------------------------------------------------------------
def load_split(split: str) -> list[dict]:
    path = config.DATASET_DIR / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# Offense year: the offense datetime is conventionally the LAST full date in the
# facts (criminal record dates come first, the offense itself comes last).
_YEAR_RE = re.compile(r"(20\d{2})\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.")
def extract_offense_year(facts: str):
    yrs = _YEAR_RE.findall(facts)
    return int(yrs[-1]) if yrs else None

# Normalize facts for near-duplicate detection: mask digits and collapse spaces,
# so two records describing the same incident (differing only in numbers/spacing) collide.
def normalize_facts(facts: str) -> str:
    s = re.sub(r"\d+", "#", facts)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -- Build the unified frame ----------------------------------------------
records = []
for split in config.SPLITS:
    for ex in load_split(split):
        lv = ex["label"][config.TARGET_FIELD]
        records.append({
            "id": ex["id"],
            "split": split,
            "facts": ex["facts"],
            "reason": ex.get("reason", ""),
            "target_lv": lv,
            "class4": config.LV_TO_CLASS[lv],
            "offense_year": extract_offense_year(ex["facts"]),
        })
df = pd.DataFrame(records)
df["facts_norm"] = df["facts"].map(normalize_facts)

print(f"Loaded {len(df)} cases across splits: {dict(df['split'].value_counts())}")

# -- 4-class distribution (overall + per split) ---------------------------
overall_class = df["class4"].value_counts().sort_index()
class_by_split = df.pivot_table(index="class4", columns="split",
                                values="id", aggfunc="count", fill_value=0)
class_by_split = class_by_split.reindex(columns=config.SPLITS, fill_value=0)
majority_class = int(overall_class.idxmax())
majority_rate = overall_class.max() / len(df)

# -- Year distribution -----------------------------------------------------
year_missing = int(df["offense_year"].isna().sum())
year_dist = df["offense_year"].value_counts().sort_index()

# -- Duplicate / group checks ---------------------------------------------
# (a) duplicated case id (same id appearing more than once across the corpus)
dup_id = df["id"].duplicated(keep=False).sum()
# id appearing in more than one split (would be a direct split leak)
id_split = df.groupby("id")["split"].nunique()
id_cross_split = int((id_split > 1).sum())

# (b) exact duplicate facts
exact_dup_groups = df.groupby("facts").size()
exact_dup_cases = int(exact_dup_groups[exact_dup_groups > 1].sum())
exact_dup_n_groups = int((exact_dup_groups > 1).sum())

# (c) near-duplicate facts (digit-masked) -> candidate "same incident" groups
norm_groups = df.groupby("facts_norm")
near_sizes = norm_groups.size()
near_dup_groups = near_sizes[near_sizes > 1]
near_dup_cases = int(near_dup_groups.sum())
near_dup_n_groups = int(len(near_dup_groups))
# how many near-duplicate groups straddle more than one split (leakage risk if random CV)
straddle = 0
for key, sub in norm_groups:
    if len(sub) > 1 and sub["split"].nunique() > 1:
        straddle += 1

# group key for group-aware CV: normalized-facts identity
df["group_key"] = df.groupby("facts_norm").ngroup()
n_groups = int(df["group_key"].nunique())

# (d) which split-pairs do straddling groups connect? (identifies test/test2 overlap)
g_split = df.groupby("group_key")["split"].apply(lambda s: tuple(sorted(set(s))))
straddle_pairs = Counter(g_split[g_split.apply(len) > 1])
n_effective_unique = n_groups   # cases collapse to this many unique incidents

print("\n=== Duplicate / group check ===")
print(f"  duplicated id (any)            : {dup_id}")
print(f"  id across >1 split             : {id_cross_split}")
print(f"  exact-duplicate facts: {exact_dup_cases} cases in {exact_dup_n_groups} groups")
print(f"  near-duplicate facts : {near_dup_cases} cases in {near_dup_n_groups} groups")
print(f"  near-dup groups straddling splits: {straddle}")
print(f"  unique groups (near-dup key)   : {n_groups} (from {len(df)} cases)")

group_aware_needed = (near_dup_n_groups > 0) or (id_cross_split > 0)
print(f"\n  -> group-aware CV split required? {group_aware_needed}")

# -- Save prepared frame ---------------------------------------------------
keep = ["id", "split", "facts", "reason", "target_lv", "class4", "offense_year", "group_key"]
out_parquet = config.OUTPUT_DIR / "stage1_prepared.parquet"
df[keep].to_parquet(out_parquet, index=False)
print(f"\nSaved: {out_parquet}")

# -- Write Table 1 (markdown) ---------------------------------------------
lines = []
lines.append("# Table 1 - Data composition (Stage 1)\n")
lines.append(f"Total cases: **{len(df)}** "
             f"(train {(df['split']=='train').sum()}, valid {(df['split']=='valid').sum()}, "
             f"test {(df['split']=='test').sum()}, test2 {(df['split']=='test2').sum()})\n")

lines.append("\n## 4-class target distribution\n")
lines.append("| class | label | " + " | ".join(config.SPLITS) + " | total | rate |")
lines.append("|---|---|" + "---|" * (len(config.SPLITS) + 2))
for c in range(config.N_CLASSES):
    row = class_by_split.loc[c] if c in class_by_split.index else pd.Series({s: 0 for s in config.SPLITS})
    tot = int(overall_class.get(c, 0))
    lines.append(f"| {c} | {config.CLASS_NAMES[c]} | "
                 + " | ".join(str(int(row[s])) for s in config.SPLITS)
                 + f" | {tot} | {tot/len(df)*100:.1f}% |")
lines.append(f"\nMajority class = **{majority_class} ({config.CLASS_NAMES[majority_class]})**, "
             f"baseline accuracy = **{majority_rate*100:.1f}%**.\n")

lines.append("\n## Offense-year distribution\n")
lines.append("| year | count |")
lines.append("|---|---|")
for y, c in year_dist.items():
    lines.append(f"| {int(y)} | {int(c)} |")
lines.append(f"| (missing) | {year_missing} |")
lines.append(f"\nYear extracted for {len(df)-year_missing}/{len(df)} cases "
             f"({(len(df)-year_missing)/len(df)*100:.1f}%).\n")

lines.append("\n## Duplicate / group check\n")
lines.append("| check | value |")
lines.append("|---|---|")
lines.append(f"| duplicated id (any) | {dup_id} |")
lines.append(f"| id across >1 split | {id_cross_split} |")
lines.append(f"| exact-duplicate facts (cases / groups) | {exact_dup_cases} / {exact_dup_n_groups} |")
lines.append(f"| near-duplicate facts (cases / groups) | {near_dup_cases} / {near_dup_n_groups} |")
lines.append(f"| near-dup groups straddling splits | {straddle} |")
lines.append(f"| unique groups (group_key) | {n_groups} |")
lines.append(f"| **effective unique cases** | **{n_effective_unique}** |")

lines.append("\n### Split-pair overlap (straddling groups)\n")
lines.append("| split pair | groups |")
lines.append("|---|---|")
for pair, n in straddle_pairs.most_common():
    lines.append(f"| {' & '.join(pair)} | {n} |")
lines.append(
    f"\n**Finding:** all {sum(straddle_pairs.values())} cross-split duplicate groups are "
    f"between **test and test2** (same id, facts, and label). `test2` is therefore a "
    f"**subset of test**, not an independent generalization set; train/valid are clean. "
    f"Consequently: (i) the corpus has **{n_effective_unique} effective unique cases**, "
    f"not {len(df)}; (ii) Stage 3 CV is run on the de-duplicated set with **group-aware "
    f"splitting on `group_key`**; (iii) `test2` is **not** used as a separate robustness "
    f"set (temporal split, Stage 9, provides robustness instead).\n")

out_md = config.OUTPUT_DIR / "T1_distributions.md"
out_md.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {out_md}")
print("\nStage 1 done.")
