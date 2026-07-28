# Codebook

Column definitions for every data file in this release. See `README.md` in this folder
for the source, licence, and the list of modifications made to the source corpus.

This codebook documents the dataset deposited at https://doi.org/10.5281/zenodo.XXXXXXX.

**Conventions**

- All files are UTF-8. CSV files carry a byte-order mark (`utf-8-sig`) so that
  spreadsheet software renders the Korean text correctly.
- CSV follows RFC 4180. The `facts`, `reason`, and `ruling_text` fields contain line
  breaks, so they appear inside quoted fields. Parsers that do not honour quoted line
  breaks will mis-read these files; the JSONL versions are the canonical form.
- Missing values are an empty field in CSV and `null` in JSON/JSONL. `-1` is not a
  missing-value code in the derived tables. It *is* used in the source corpus's
  `ruling.parse` fields, where it means "no penalty of this kind was imposed".
- Case `id` is the identifier assigned by the source corpus and is stable across every
  file here, so tables can be joined on it.

---

## 1. `raw/` — the drunk-driving subset of the source corpus

| file | records | notes |
|---|---|---|
| `dui_cases_1644.jsonl` | 1,644 | original nested structure, one JSON object per line |
| `dui_cases_1644.csv` | 1,644 | same records, nested fields flattened |
| `dui_cases_1500_unique.jsonl` | 1,500 | de-duplicated analysis set, plus `class4` / `group_key` |
| `dui_cases_1500_unique.csv` | 1,500 | flattened form of the above |

**1,644 vs 1,500.** The source corpus ships four splits. Its `test2` split is a strict
subset of `test`: all 144 records share an `id`, a `facts` text, and a label with a record
already in `test`. Verified independently by normalised-text grouping. `dui_cases_1644.*`
therefore repeats 144 cases and **`id` is not unique in it**; it is included only so the
source splits can be reconstructed exactly. Every analysis in the paper runs on the 1,500
unique cases (train 1,200 / valid 150 / test 150), which is what
`dui_cases_1500_unique.*` contains.

### Columns

| column | type | description |
|---|---|---|
| `id` | integer | case identifier from the source corpus |
| `split` | string | `train` / `valid` / `test` / `test2` — the source corpus split (`test2` only in the 1,644-record files) |
| `casetype` | string | always `criminal` |
| `casename` | string | always `도로교통법위반(음주운전)` (Road Traffic Act violation, drunk driving) — the filter that defines this subset |
| `facts` | string | the criminal facts section of the judgment. Model input. Contains line breaks |
| `reason` | string | the sentencing-reasons section. Not used as model input |
| `class4` | integer | **derived.** 4-class ordinal target, 0–3. Unique-set files only |
| `class4_name` | string | **derived.** `fine/none` / `short` / `medium` / `long`. Unique-set files only |
| `group_key` | integer | **derived.** Index of the normalised-`facts` identity group, used for group-aware cross-validation so that duplicated incidents cannot straddle a fold boundary. Unique-set files only |

Nested fields, and their names in the flattened CSV:

| JSONL path | CSV column | type | description |
|---|---|---|---|
| `label.text` | `label_text` | string | the sentence as written, e.g. `징역 12월` |
| `label.fine_lv` | `fine_lv` | integer | source corpus fine level, 0–4 |
| `label.imprisonment_with_labor_lv` | `imprisonment_with_labor_lv` | integer | source corpus imprisonment-with-labour level. **The prediction target of this study.** Range 0–4 in this subset |
| `label.imprisonment_without_labor_lv` | `imprisonment_without_labor_lv` | integer | imprisonment-without-labour level, 0–4 |
| `ruling.text` | `ruling_text` | string | the disposition (주문) as written |
| `ruling.parse.fine.type` | `ruling_fine_type` | string | `벌금` or empty |
| `ruling.parse.fine.unit` | `ruling_fine_unit` | string | currency unit or empty |
| `ruling.parse.fine.value` | `ruling_fine_value` | integer | fine in KRW; `-1` = no fine imposed |
| `ruling.parse.imprisonment.type` | `ruling_imprisonment_type` | string | `징역` / `집행유예` / `금고` / empty |
| `ruling.parse.imprisonment.unit` | `ruling_imprisonment_unit` | string | `mo` (months) or empty |
| `ruling.parse.imprisonment.value` | `ruling_imprisonment_value` | integer | sentence length in months; `-1` = no custodial sentence |

`ruling` and `reason` are provided for transparency. Neither is a model input in this
study: the models see `facts` only, which is why the disposition text cannot leak the
answer. This was audited separately (`results/Sec3-1_leakage_audit.csv`).

### Target construction

`class4` collapses the source corpus's `imprisonment_with_labor_lv` (0–4 in this subset)
into four ordinal classes, merging levels 1 and 2 because both denote short custodial
terms and level 1 alone is sparse:

| `imprisonment_with_labor_lv` | `class4` | name | approximate sentence | n (of 1,500) |
|---|---|---|---|---|
| 0 | 0 | `fine/none` | fine or no imprisonment | 215 |
| 1, 2 | 1 | `short` | 3–10 months | 107 |
| 3 | 2 | `medium` | 12–17 months | 1,017 |
| 4 | 3 | `long` | 18–30 months | 161 |

The `medium` class is 67.8% of the data. A majority-class baseline is reported throughout
the paper for this reason, and accuracy is never used as a sole metric.

---

## 2. `benchmark/` — the attorney-verified feature table

`benchmark_features.csv` / `.json` — 1,500 rows, one per unique case. These
five columns are the authors' own annotation: rule-extracted from `facts`, then reviewed
case by case by a licensed Korean attorney, who corrected 114 of 7,500 values (98.5%
initial accuracy). Per-feature accuracy of the automatic extraction before correction:
BAC 99.5%, distance 99.3%, vehicle 93.9%, prior 99.7%, year 99.9%.

| column | type | description | missing |
|---|---|---|---|
| `id` | integer | case identifier | 0 |
| `split` | string | `train` (1,200) / `valid` (150) / `test` (150) | 0 |
| `class4` | integer | 4-class ordinal target, 0–3 (see above) | 0 |
| `bac` | float | blood alcohol concentration, percent by volume (e.g. `0.126` = 0.126%). Observed range 0.003–0.352. Where several readings appear for one defendant, the highest is kept | 3 |
| `distance_km` | float | driving distance in kilometres. Observed range 0.001–130 | 37 |
| `vehicle` | string | `passenger` (1,208) / `truck` (183) / `van` (53) / `motorcycle` (47) / `other` (9) | 0 |
| `prior_dui_count` | integer | number of prior drunk-driving convictions, counted from dated events in the record section. Range 0–8 | 0 |
| `offense_year` | integer | calendar year of the offence, 2007–2022. Used for the temporal-split robustness analysis (train < 2021, test ≥ 2021) | 0 |

Missing `bac` and `distance_km` are genuinely absent from the judgment text, not
extraction failures; they are handled by imputation fitted inside each training fold.

`splits.csv` — 1,500 rows.

| column | type | description |
|---|---|---|
| `id` | integer | case identifier |
| `split` | string | `train` / `valid` / `test` |
| `group_key` | integer | normalised-`facts` identity group, for group-aware cross-validation |

---
