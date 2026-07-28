# Korean DUI Sentencing Dataset: an attorney-verified benchmark of 1,500 cases

A drunk-driving (DUI) subset of the Korean LBox Open judgment corpus, de-duplicated, recoded
to a four-class ordinal sentencing target, and annotated with five sentencing factors that a
licensed Korean attorney verified case by case.

This folder is self-contained and is deposited under its own DOI,
https://doi.org/10.5281/zenodo.XXXXXXX, separately from the experiments that use it.
The analysis code, model predictions, and statistical results live in
`../2_experiment/`.

## Description

Each record is one Korean criminal judgment for drunk driving under Article 148-2 of the
Road Traffic Act. The prediction target is the imprisonment severity the court imposed, on a
four-class ordinal scale:

| class | name | approximate sentence | n (of 1,500) |
|---|---|---|---|
| 0 | `fine/none` | fine or no imprisonment | 215 |
| 1 | `short` | 3–10 months | 107 |
| 2 | `medium` | 12–17 months | 1,017 |
| 3 | `long` | 18–30 months | 161 |

The medium class is 67.8% of the data, so any evaluation on this dataset should report a
majority-class baseline and should not rely on accuracy alone.

Alongside the judgment text, five legally salient factors are provided as structured
columns: blood alcohol concentration, driving distance, vehicle type, number of prior DUI
convictions, and the year of the offence. These were extracted by a frozen rule set and then
reviewed in full by a licensed Korean attorney, who corrected 114 of the 7,500 extracted
values, so the released table is verified rather than automatically labelled. Per-factor
accuracy of the automatic extraction before correction was: BAC 99.5%, distance 99.3%,
vehicle 93.9%, prior convictions 99.7%, year 99.9%.

## Files

| file | records | contents |
|---|---|---|
| `raw/dui_cases_1644.jsonl` | 1,644 | all four source splits, original nested structure, one JSON object per line |
| `raw/dui_cases_1644.csv` | 1,644 | the same records with nested fields flattened |
| `raw/dui_cases_1500_unique.jsonl` | 1,500 | the de-duplicated analysis set, plus the four-class target, its name, and the grouping key |
| `raw/dui_cases_1500_unique.csv` | 1,500 | the flattened form of the above |
| `benchmark/benchmark_features.csv` | 1,500 | the five verified factors and the four-class label |
| `benchmark/benchmark_features.json` | 1,500 | the same table as JSON records |
| `benchmark/splits.csv` | 1,500 | `id`, `split`, and `group_key` for group-aware evaluation |

**`codebook.md` defines every column of every file above.** Read it first.

**1,644 versus 1,500.** The source corpus ships four splits, and its `test2` split is a
strict subset of `test`: all 144 records share an `id`, a `facts` text, and a label with a
record already in `test`. The 1,644-record files therefore repeat 144 cases and `id` is not
unique in them; they are included only so the source splits can be reconstructed exactly.
Use the 1,500-case files for any analysis.

**Grouping.** `group_key` indexes cases whose normalised facts text is identical. Use it to
keep duplicated incidents from straddling a fold boundary; ignoring it inflates
cross-validation scores.

## Usage

```python
import json
import pandas as pd

factors = pd.read_csv("benchmark/benchmark_features.csv", encoding="utf-8-sig")
splits = pd.read_csv("benchmark/splits.csv")

with open("raw/dui_cases_1500_unique.jsonl", encoding="utf-8") as fh:
    cases = [json.loads(line) for line in fh]

# The judgment text and the structured factors join on the case id.
text = pd.DataFrame([{"id": c["id"], "facts": c["facts"]} for c in cases])
data = factors.merge(text, on="id").merge(splits[["id", "group_key"]], on="id")
```

CSV files carry a UTF-8 byte-order mark so that spreadsheet software renders the Korean text
correctly. The `facts`, `reason`, and `ruling_text` fields contain line breaks and therefore
sit inside quoted fields; parsers that do not honour quoted line breaks will mis-read the CSV,
so the JSONL versions are the canonical form.

The only requirement for loading the data is a CSV or JSON reader; `pandas` is convenient but
not necessary.

## Source, licence, and modifications

This dataset is a filtered, de-duplicated, and re-annotated subset of **LBox Open**
(`lbox_open`), task `ljp_criminal`, created by **LBox Co., Ltd.**

- Source: https://github.com/lbox-kr/lbox-open · https://huggingface.co/datasets/lbox/lbox_open
- Licence: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** —
  https://creativecommons.org/licenses/by-nc/4.0/

**Changes made to the source material** (indicated as required by CC BY-NC 4.0 §3(a)(1)(B)):

1. Filtered to `casename == "도로교통법위반(음주운전)"`, yielding 1,644 records.
2. Removed 144 duplicated records, leaving 1,500 unique cases.
3. Recoded `label.imprisonment_with_labor_lv` (0–4 in this subset) into a four-class ordinal
   target, merging levels 1 and 2.
4. Added five feature columns created by the authors, rule-extracted and then verified case
   by case by a licensed Korean attorney.
5. Added a `group_key` grouping index derived from normalised `facts`.
6. The `facts`, `reason`, and `ruling` text of retained records is reproduced verbatim.

This derivative is redistributed under the same licence, **CC BY-NC 4.0, for non-commercial
research use only**. See `LICENSE` in this folder.

The five derived feature columns and the four-class label mapping are the authors' own work.
Taken alone and without any source text — the contents of `benchmark/` — they are
additionally offered under CC BY 4.0.

## Citation

Please cite both this dataset and the source corpus.

> Choi, S., & Oh, T. (2026). *Korean DUI Sentencing Dataset: an attorney-verified
> benchmark of 1,500 cases* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

> Hwang, W., Lee, D., Cho, K., Lee, H., & Seo, M. (2022). A Multi-Task Benchmark for Korean
> Legal Language Understanding and Judgement Prediction. *Advances in Neural Information
> Processing Systems 35*, Datasets and Benchmarks Track. arXiv:2206.05224

If you use the benchmark as it was used in the accompanying study, please also cite the paper
(citation to be added on publication).

## Acknowledgement

We thank LBox Co., Ltd. for creating and openly releasing the LBox Open corpus, and the legal
reviewer who verified the extracted sentencing factors.
