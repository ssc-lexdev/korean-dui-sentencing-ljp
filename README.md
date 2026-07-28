# Korean DUI Legal Judgment Prediction: Classical ML, LLMs, and Explanation Agreement

Reproduction materials for the paper:

> *Reproducing and explaining legal judgment prediction on Korean drunk-driving cases:
> a benchmark of classical machine learning, large language models, and cross-model
> explanation agreement.*

- [Description](#description)
- [Dataset information](#dataset-information)
- [Code information](#code-information)
- [Usage instructions](#usage-instructions)
- [Requirements](#requirements)
- [Methodology](#methodology)
- [Citation](#citation)
- [Licence](#licence)
- [Contributing](#contributing)

---

## Description

The study predicts the sentence a Korean court imposed in a drunk-driving (DUI) case from
the criminal facts of the judgment, as a **four-class ordinal** target: 0 = fine or no
imprisonment, 1 = short term, 2 = medium term, 3 = long term. It compares two families of
models on the same 1,500 cases, on both what they predict and how they explain it:

- **Classical machine learning** — SVM, logistic regression, random forest, and LightGBM,
  each over two input representations: five lawyer-verified sentencing factors, and TF-IDF
  over the raw facts. Eight configurations, plus majority-class and random baselines.
- **Large language models** — a proprietary model (GPT-5) and an open-weight model
  (Qwen3-8B-Instruct), each zero-shot closed-book and few-shot open-book. Four
  configurations.

The labels are heavily imbalanced: the medium class is about 68% of the data. A
majority-class baseline is therefore reported throughout, and accuracy is never used on its
own — macro-F1 and quadratic-weighted kappa (QWK) are the primary metrics. Reading accuracy
alone inverts the ranking of the systems, which is one of the paper's points.

This repository contains everything needed to reproduce every reported number: the data in
open formats, the analysis code as it was run, the model predictions and cached LLM
replies, the statistical result tables, and the figures, tables, and supplement.

Folders are numbered so they list in the order the work flows. The dataset is kept separate
from everything derived from it, and is deposited under its own DOI.

```
1_dataset/                            the dataset the study is built on
  raw/                                the drunk-driving subset, JSONL + CSV
  benchmark/                          the attorney-verified factors and splits
  README.md, codebook.md, LICENSE     self-contained; deposited separately

2_experiment/
  2-1_experiment_code/                the analysis pipeline
  2-2_experiment_result/              everything the pipeline produced
    predictions/                      ML out-of-fold and LLM predictions
    llm_caches/                       the models' replies, verbatim
    results/                          the statistical result tables
    codebook.md                       column definitions for the three above

3_tables/                             Tables 1-4 (main text)
4_figures/                            Figures 1-3 (main text) + CAPTIONS.txt
5_supplement/                         supplementary text and tables, the prompt
                                      document, Figures S1-S2
```

---

## Dataset information

The dataset the study is built on is in `1_dataset/`, in an open, readable format. It is
**self-contained and separately citable**: it carries its own `README.md`, `codebook.md`, and
`LICENSE`, and is deposited under its own DOI so that it can be reused and referenced
independently of these experiments. `1_dataset/README.md` is the fuller description;
`1_dataset/codebook.md` defines every column.

| path | contents |
|---|---|
| `1_dataset/raw/dui_cases_1644.{jsonl,csv}` | the drunk-driving subset of the source corpus, all four source splits. `id` is **not** unique here: the source `test2` split repeats 144 cases from `test` |
| `1_dataset/raw/dui_cases_1500_unique.{jsonl,csv}` | the de-duplicated analysis set actually used in the paper, with the four-class target, the class name, and the grouping key attached |
| `1_dataset/benchmark/benchmark_features.{csv,json}` | the 1,500-case benchmark: the five sentencing factors used as structured inputs (BAC, driving distance, vehicle type, prior-DUI count, offence year) and the four-class label |
| `1_dataset/benchmark/splits.csv` | group-aware split definitions (`id`, `split`, `group_key`), train 1,200 / valid 150 / test 150 |

The five feature columns are the authors' own annotation: rule-extracted from the facts
text, then reviewed case by case by a licensed Korean attorney, who corrected 114 of 7,500
values (98.5% initial accuracy).

CSV files carry a UTF-8 byte-order mark so spreadsheet software renders the Korean text
correctly. The `facts`, `reason`, and `ruling_text` fields contain line breaks, so they sit
inside quoted fields; the JSONL versions are the canonical form. Parquet copies are kept
for convenience only and duplicate the CSV exactly.

Everything produced *from* this dataset — model predictions, cached LLM replies, and the
statistical result tables — is in `2_experiment/2-2_experiment_result/` and is described
under [Code information](#code-information).

### Source, licence, and modifications

The case data are a subset of **LBox Open** (`lbox_open`), task `ljp_criminal`, created by
**LBox Co., Ltd.**

- Source: https://github.com/lbox-kr/lbox-open · https://huggingface.co/datasets/lbox/lbox_open
- Licence: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** —
  https://creativecommons.org/licenses/by-nc/4.0/
- Citation: Hwang, W., Lee, D., Cho, K., Lee, H., & Seo, M. (2022). *A Multi-Task Benchmark
  for Korean Legal Language Understanding and Judgement Prediction.* Advances in Neural
  Information Processing Systems 35, Datasets and Benchmarks Track. arXiv:2206.05224

**Changes made to the source material** (indicated as required by CC BY-NC 4.0 §3(a)(1)(B)):

1. Filtered to `casename == "도로교통법위반(음주운전)"`, yielding 1,644 records.
2. Removed 144 duplicated records. The source `test2` split is a strict subset of `test`
   (same `id`, `facts`, and label), leaving 1,500 unique cases.
3. Recoded `label.imprisonment_with_labor_lv` (0–4 in this subset) into a four-class ordinal
   target, merging levels 1 and 2.
4. Added five feature columns created by the authors — BAC, driving distance, vehicle type,
   prior-DUI count, offence year — rule-extracted from `facts` and then manually verified by
   a licensed Korean attorney.
5. Added a `group_key` grouping index derived from normalised `facts`.
6. The `facts`, `reason`, and `ruling` text of retained records is reproduced verbatim. No
   text was altered, and nothing was added to or removed from the text of any record.

This derivative is redistributed under the same licence, **CC BY-NC 4.0, for non-commercial
research use only**. See [Licence](#licence).

---

## Code information

All analysis code is in `2_experiment/2-1_experiment_code/`. The stage scripts are published **exactly as they were
run**, so they use the pipeline's internal file names rather than the publication names in
`1_dataset/`; `prepare_workspace.py` reconciles the two (see [Usage
instructions](#usage-instructions)).

### Shared modules and setup

| file | what it does |
|---|---|
| `config.py` | Paths, the global seed (42), the level-to-class mapping, the feature list, model and metric definitions. Every script imports it. Run it directly to print the resolved configuration as a self-check |
| `features.py` | The frozen rule-based extractor for the five sentencing factors from Korean facts text |
| `prepare_workspace.py` | **Run first.** Stages `1_dataset/` into the layout the stage scripts expect, and maps publication file names to pipeline names |
| `law_context.txt` | The statute and sentencing-guideline text used as the open-book context, byte-identical to the string sent to the models. Read by `stage6_llm.py` |
| `make_data_release_files.py` | Rebuilds the open-format files in `1_dataset/` (CSV / JSON / JSONL) from a local copy of the subset |

### Pipeline stages

Run in this order. Each writes to the working directory that `prepare_workspace.py` sets up.

| file | stage | what it does | supports |
|---|---|---|---|
| `stage1_data_prep.py` | 1 | Loads the four source splits, applies the four-class relabelling, extracts the offence year, and runs the duplicate and group check that establishes `test2` ⊂ `test` | §3.1 |
| `stage2_features.py` | 2 | Applies the frozen extractor to the facts text, reports per-factor coverage, and emits the sample used for manual validation | §3.1 |
| `stage_disposition_audit.py` | — | Leakage audit: checks whether the retained `facts` text ever contains the court's current disposition, which would leak the label into the TF-IDF models | §3.1 |
| `stage3_ml.py` | 3 | The classical-ML benchmark: 4 models × {structured, TF-IDF} plus two baselines, under repeated stratified group-aware 5×5 cross-validation, with all preprocessing fit inside the training fold | Table 2, §4.1 |
| `stage4_stats.py` | 4 | Tests each configuration against the majority baseline on macro-F1 and QWK, with bootstrap and repeat-clustered mixed-effects intervals, Holm correction, and Cohen's *d* | §4.1 (RQ1) |
| `stage5_shap.py` | 5 | TreeSHAP on the LightGBM model over the five verified factors: global importance, per-class beeswarm, and a local waterfall. Produces the ML explanation ranking *r*<sub>ML</sub> | Figure 3, §4.4 |
| `stage6_llm.py` | 6 | LLM inference for all four configurations. Every reply is cached by case `id`, so a rerun with the released caches makes no API call | §3.3, §4.2 |
| `stage7_h2.py` | 7 | Paired McNemar of the best ML model against each LLM configuration on the same 1,500 cases, under exact-match and ordinal (±1) correctness, Holm-corrected within each family | Table 3, §4.2 |
| `stage7b_qwk_bootstrap.py` | 7b | Case-bootstrap 95% confidence interval for the ML − LLM difference in QWK | §4.2 (RQ2) |
| `stage8_agreement.py` | 8 | Compares *r*<sub>ML</sub> against each LLM's factor ranking in the shared five-factor space, by citation frequency and by order of mention, with Kendall's tau and top-set Jaccard | Table 4, §4.4 |
| `stage8b_tau_kernelshap.py` | 8b | Robustness of the agreement result: bootstrap interval for the order-of-mention tau, and a model-agnostic KernelSHAP ranking on the structured SVM as a cross-check on TreeSHAP | §4.4 (RQ3) |
| `stage9_temporal.py` | 9 | Temporal robustness: refits the same pipelines on a train-on-earlier, test-on-later split and compares against the cross-validation figures | §4.5 |
| `stage9b_temporal_ablation.py` | 9b | Factor ablations on the temporal split, isolating how much of the surviving signal is carried by the prior-DUI count | §4.5, §5.4 |

### Figure scripts

| file | what it does |
|---|---|
| `make_fig2_error_geometry.py` | Figure 2: row-normalised confusion matrices and predicted-share strips for the best ML model and the two zero-shot LLMs. Writes `4_figures/Figure_2.png` |
| `make_supp_shap_figs.py` | Figures S1 and S2: SHAP beeswarm and local waterfall. Writes into `5_supplement/` |

`stage5_shap.py` also writes `4_figures/Figure_3.png`. Figure 1 is a hand-drawn pipeline
diagram and has no generating script.

`2_experiment/2-1_experiment_code/` holds only what the paper needs. The two scripts that
turned the attorney's review sheet into the verified feature table are not included: their
inputs are a human review workflow rather than data, and their output is released directly
as `1_dataset/benchmark/benchmark_features.csv`.

### What the pipeline produced

Everything the code wrote is in `2_experiment/2-2_experiment_result/`, so the reported
numbers can be inspected without re-running anything.
`2_experiment/2-2_experiment_result/codebook.md` defines every column.

| path | contents |
|---|---|
| `predictions/ml_oof_predictions.csv` | classical-ML out-of-fold predictions, 1,500 cases × 8 configurations |
| `predictions/llm_predictions.{csv,jsonl}` | LLM predictions, 1,500 cases × 4 configurations, including the factors each model said it relied on |
| `llm_caches/*.json` | the models' replies verbatim, keyed by case `id`; these reproduce every LLM number without calling an API |
| `results/*.csv` | the statistical result tables, each named for the manuscript section, table, or figure it supports, so the folder sorts in manuscript order |

---

## Usage instructions

### Quick start

```bash
git clone https://github.com/ssc-lexdev/korean-dui-sentencing-ljp.git
cd korean-dui-sentencing-ljp

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python 2_experiment/2-1_experiment_code/prepare_workspace.py    # required — see below
python 2_experiment/2-1_experiment_code/stage1_data_prep.py     # then the rest, in the order listed above
```

### Loading the data on its own

The data files stand alone; you do not need to run the pipeline to use them.

```python
import json
import pandas as pd

# The five verified sentencing factors and the four-class label, 1,500 rows.
factors = pd.read_csv("1_dataset/benchmark/benchmark_features.csv", encoding="utf-8-sig")

# The case texts. JSONL is canonical: the facts field contains line breaks.
with open("1_dataset/raw/dui_cases_1500_unique.jsonl", encoding="utf-8") as fh:
    cases = [json.loads(line) for line in fh]

# Model predictions, and the result table behind Table 2 of the paper.
ml = pd.read_csv("2_experiment/2-2_experiment_result/predictions/ml_oof_predictions.csv", encoding="utf-8-sig")
llm = pd.read_csv("2_experiment/2-2_experiment_result/predictions/llm_predictions.csv", encoding="utf-8-sig")
table2 = pd.read_csv("2_experiment/2-2_experiment_result/results/Sec4-1_Table2_ml_performance_per_fold.csv")

# Everything joins on the case id.
merged = factors.merge(llm[llm.variant == "few_shot_open_book"], on="id")
```

### Why `prepare_workspace.py` is required

The stage scripts are published exactly as they were run, so they read from
`config.DATASET_DIR` and `config.OUTPUT_DIR` — the authors' working layout — using the
pipeline's internal file names. Running a stage script on a fresh clone without staging
first fails with `FileNotFoundError`. The script does the mapping:

| pipeline expects | staged from |
|---|---|
| `<DATASET_DIR>/{train,valid,test,test2}.jsonl` | `1_dataset/raw/dui_cases_1644.jsonl`, split back out |
| `stage2_features_verified.parquet` | `1_dataset/benchmark/benchmark_features.csv` |
| `stage3_oof_predictions.parquet` | `2_experiment/2-2_experiment_result/predictions/ml_oof_predictions.csv` |
| `stage6_llm_predictions.parquet` | `2_experiment/2-2_experiment_result/predictions/llm_predictions.csv` |
| `llm_cache_<model>_<variant>.json` | `2_experiment/2-2_experiment_result/llm_caches/` (names unchanged) |
| `stage3_cv_results.csv` and the other result tables | `2_experiment/2-2_experiment_result/results/`, whose files are named for the manuscript item they support (`RESULT_TABLES` in the script holds the mapping) |

`stage1_prepared.parquet` is not staged: `stage1_data_prep.py` builds it from the staged
source splits, which is why that script must run first.

### Re-running the LLM stage

`stage6_llm.py` reads API keys **only** from the environment variables `OPENAI_API_KEY` and
`OPENROUTER_API_KEY`. No keys are stored in this repository.

With the released caches in place the script makes **no API call** — every case is already
cached, so it re-parses the stored replies and reproduces the reported numbers offline. Both
environment variables must still be set to something, because the clients are constructed at
start-up; any placeholder value works.

```bash
OPENAI_API_KEY=unused OPENROUTER_API_KEY=unused python 2_experiment/2-1_experiment_code/stage6_llm.py
```

Deleting a cache file makes the script call the provider for every case in that
configuration, which costs money. We claim deterministic decoding and cached predictions for
re-use, not exact re-generation: provider-side routing limits the exact reproducibility of a
proprietary model, so model versions and call dates are recorded in the caches and in the
paper.

`stage6_llm.py` builds its open-book context from `2_experiment/2-1_experiment_code/law_context.txt`. Sections 2 and 3
of `5_supplement/few_shot_open_book_prompt.md` are a condensed reading version of the same
material, so build the prompt from `law_context.txt`, not from that document.

---

## Requirements

- **Python 3.13** (developed on 3.13.9; also verified on 3.14).
- **No GPU is required.** Every stage runs on CPU. The two LLMs are reached over HTTP APIs,
  so no local model weights are downloaded and no CUDA device is needed.
- Install everything with `pip install -r requirements.txt` rather than picking packages by
  hand. Several stages fail on a dependency that is easy to overlook:

| package | needed by |
|---|---|
| numpy, pandas, scipy, pyarrow | all stages |
| scikit-learn | stages 3, 5, 8b, 9, 9b |
| lightgbm, xgboost | stages 3, 5, 8b, 9, 9b |
| statsmodels | stages 4 and 7 (Holm correction, mixed-effects intervals, McNemar) |
| shap, lime | stages 5 and 8b, and the supplementary figure script |
| matplotlib | stage 5 and both figure scripts |
| openai | stage 6 only |

Versions are pinned in `requirements.txt`. `ENVIRONMENT.md` records the exact machine, the
seed and cross-validation settings, and the leakage-control rules.

Runtime is dominated by `stage3_ml.py`, which fits eight configurations across 25 folds,
including TF-IDF SVMs. Everything else finishes quickly, and `stage6_llm.py` is near-instant
when run against the caches.

---

## Methodology

The pipeline runs in nine stages, in this order.

1. **Data preparation.** Load the drunk-driving subset, apply the four-class relabelling,
   and extract the offence year. A duplicate and group check establishes that the source
   `test2` split is a strict subset of `test`, so the corpus holds 1,500 unique cases rather
   than 1,644. A `group_key` derived from normalised facts text is attached so that
   duplicated incidents cannot straddle a cross-validation fold boundary.
2. **Feature extraction and expert verification.** A frozen rule-based extractor reads five
   sentencing factors from the facts text. A licensed Korean attorney then reviewed all
   1,500 cases and corrected 114 of the 7,500 extracted values. The corrected table is the
   ground truth used everywhere downstream. A separate leakage audit confirms the retained
   facts text never states the court's current disposition.
3. **Classical machine learning.** Four models over two representations, under repeated
   stratified group-aware 5×5 cross-validation. All preprocessing — imputation, scaling,
   one-hot encoding, TF-IDF fitting — happens inside each training fold. Hyperparameters are
   library defaults, fixed a priori rather than tuned, so the classical models are an untuned
   baseline.
4. **Statistical testing.** Each configuration is compared against the majority-class
   baseline on macro-F1 and QWK, with bootstrap and repeat-clustered mixed-effects intervals,
   Holm correction across the family of tests, and Cohen's *d*.
5. **Explanation of the ML model.** TreeSHAP over the five verified factors yields the global
   importance ranking *r*<sub>ML</sub>.
6. **LLM inference.** Two models, each zero-shot closed-book and few-shot open-book. The
   open-book prompt adds the statute, the sentencing guideline, and four class-balanced
   worked examples drawn from the training split. Decoding is deterministic and every reply
   is cached. Each model returns a class label and the factors it relied on, under a fixed
   output format; a regular-expression protocol parses both.
7. **Prediction comparison.** The best ML model is paired against each LLM configuration on
   the same 1,500 cases, using McNemar under exact-match and ordinal (±1) correctness, plus a
   case-bootstrap interval for the QWK difference.
8. **Explanation comparison.** Each LLM's stated factors are mapped into the same five-factor
   space as *r*<sub>ML</sub>, giving two signals: how often a factor is cited, and how early
   it is mentioned. The two rankings are compared with Kendall's tau and top-set Jaccard,
   with a permutation test, a bootstrap interval, and a KernelSHAP cross-check.
9. **Temporal robustness.** The same pipelines are refit on a train-on-earlier,
   test-on-later split, with factor ablations to see which factors carry the surviving
   signal across a change in the legal regime.

---

## Citation

Please cite the paper (citation to be added on publication), the dataset, and the source
corpus.

> Choi, S., & Oh, T. (2026). *Korean DUI Sentencing Dataset: an attorney-verified
> benchmark of 1,500 cases* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21647662

> Hwang, W., Lee, D., Cho, K., Lee, H., & Seo, M. (2022). A Multi-Task Benchmark for Korean
> Legal Language Understanding and Judgement Prediction. *Advances in Neural Information
> Processing Systems 35*, Datasets and Benchmarks Track. arXiv:2206.05224

This repository is also archived at Zenodo under its own DOI, which is minted when a
release is tagged and is shown on the repository page from then on. Cite that DOI when
referring to a specific version of the code, and the dataset DOI above when referring to
the data.

---

## Licence

Two licences apply, and `LICENSE` gives the full text of both.

- **Code** (`2_experiment/2-1_experiment_code/`) — MIT License.
- **Data** (`1_dataset/`) — Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0), inherited from the source corpus. Non-commercial use only, attribution
  required, modifications indicated. See [Source, licence, and
  modifications](#source-licence-and-modifications).

The five derived feature columns and the four-class label mapping are the authors' own work.
Taken alone and without any source text — the contents of `1_dataset/benchmark/` — they are
additionally offered under CC BY 4.0.

Figures, tables, and the supplement accompany the article and follow the journal's terms.

---

## Contributing

This repository is a fixed record of the analysis behind a published paper, not an actively
developed project. The code is archived so that the reported numbers can be checked, so we do
not accept changes that would alter what was run.

- **Reproduction problems** — please open an issue describing your platform, Python version,
  and the exact command and error. Reports that a stage does not run, or returns numbers that
  differ from `2_experiment/2-2_experiment_result/results/`, are the most useful thing you can send.
- **Corrections** — pull requests are welcome for documentation errors, broken links, and
  packaging or portability fixes that do not change any result.
- **Extensions** — please fork rather than open a pull request. If you build on the benchmark
  we would like to hear about it, and the licence terms above apply to any redistribution.

Questions about the paper itself are best directed to the corresponding author.

## Acknowledgement

We thank LBox Co., Ltd. for creating and openly releasing the LBox Open corpus, without which
this study would not have been possible, and the legal reviewer who verified the extracted
sentencing factors.
