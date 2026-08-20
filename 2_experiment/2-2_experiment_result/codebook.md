# Codebook — experiment results

Column definitions for the model outputs and statistical result tables in this folder.
The dataset these were produced from is documented in `../../1_dataset/codebook.md`.

**Conventions**

- All files are UTF-8. CSV files carry a byte-order mark (`utf-8-sig`) so that
  spreadsheet software renders the Korean text correctly.
- Missing values are an empty field in CSV and `null` in JSON/JSONL.
- Case `id` is the identifier assigned by the source corpus and is stable across every
  file in this repository, so tables can be joined on it.

---

## 1. `predictions/` — model outputs

`ml_oof_predictions.csv` — 12,000 rows = 1,500 cases × 8 configurations.
Out-of-fold predictions from repeated stratified group-aware 5×5 cross-validation.

| column | type | description |
|---|---|---|
| `id` | integer | case identifier |
| `model` | string | `svm` / `logreg` / `random_forest` / `lightgbm` |
| `rep` | string | input representation: `structured` (the five factors) or `tfidf` (bag of n-grams over `facts`) |
| `y_true` | integer | true `class4` |
| `y_pred` | integer | predicted `class4` |

`llm_predictions.csv` / `.jsonl` — 6,000 rows = 1,500 cases × 4 configurations.

| column | type | description |
|---|---|---|
| `id` | integer | case identifier |
| `model` | string | `gpt` (GPT-5, closed weights) or `qwen` (Qwen3-8B-Instruct, open weights) |
| `variant` | string | `zero_shot_closed_book` (facts only) or `few_shot_open_book` (facts plus the statute, the sentencing guideline, and four worked examples) |
| `y_true` | integer | true `class4` |
| `y_pred` | integer | predicted `class4`; parsed from the model reply, 100% parse success across all 6,000 calls |
| `factors` | string | the factors the model itself said it relied on, in Korean, in the order it stated them. Order of mention is the signal used for the explanation-agreement analysis |

---

## 2. `llm_caches/` — raw model replies

Four JSON files, one per configuration, named `llm_cache_<model>_<variant>.json`. Each is
a flat object keyed by case `id` **as a string**, with the value `{"raw": "<reply>"}`
holding the model's reply verbatim, e.g.:

```json
{ "4525": { "raw": "class: 2\nfactors: 음주 전과(동종 1회), 혈중알코올농도 0.252%(매우 높음), ..." } }
```

The models were prompted in Korean and answered in Korean, so the cached replies are Korean;
they are stored verbatim because they are the raw experimental output. The example above reads:
`class: 2` / `factors: prior drunk-driving convictions (1 of the same kind), blood alcohol
concentration 0.252% (very high), ...`.

These make every LLM number in the paper reproducible without re-calling the APIs. They
contain the models' restated factors, not source judgment text.

---

## 3. `results/` — statistical result tables

All CSV. One row per reported comparison; these are the numbers behind the tables and
figures in the paper.

Each file is named for the part of the manuscript it supports, so the list sorts in
manuscript order: the section, then the table or figure number where there is one. The
analysis scripts use the pipeline's own names internally; `2_experiment/2-1_experiment_code/prepare_workspace.py`
holds the mapping between the two, in `RESULT_TABLES`.

| file | rows | supports | what it holds |
|---|---|---|---|
| `Sec3-1_leakage_audit.csv` | 1 | §3.1 | leakage audit of the 1,500 `facts` texts: current-disposition matches and 주문 (*jumun*, the holding heading) matches found |
| `Sec4-1_Table2_ml_performance_per_fold.csv` | 250 | Table 2, §4.1 | per-fold ML metrics (accuracy, macro-F1, QWK, balanced accuracy, MAE in levels and months) for 8 configurations × 25 folds. Table 2 reports the mean and standard deviation over the 25 folds |
| `Sec4-1_h1_significance_tests.csv` | 16 | §4.1 (RQ1) | each model-representation against the majority baseline, on macro-F1 and QWK, with bootstrap and mixed-effects confidence intervals, Holm-adjusted p, Cohen's d |
| `Sec4-2_Table3_ml_vs_llm_mcnemar.csv` | 8 | Table 3, §4.2 | McNemar, best ML against each LLM configuration, under exact-match and ordinal (±1) correctness |
| `Sec4-2_qwk_difference_bootstrap.csv` | 4 | §4.2 (RQ2) | case-bootstrap confidence intervals (B = 10,000) for the ML − LLM difference in QWK |
| `Sec4-4_Figure3_shap_factor_ranking.csv` | 5 | Figure 3, §4.4 | TreeSHAP global importance and rank for the five factors (*r*<sub>ML</sub>) |
| `Sec4-4_Table4_explanation_agreement.csv` | 4 | Table 4, §4.4 | explanation agreement per LLM configuration: Kendall's tau and Jaccard@2/@3, for both the citation-frequency and the order-of-mention signal, with permutation p |
| `Sec4-4_llm_factor_rankings.csv` | 4 | §4.4 (RQ3) | per-factor citation percentage, mean first-mention position, and the resulting ranks (*r*<sub>LLM</sub>) |
| `Sec4-4_tau_bootstrap_kernelshap.csv` | 6 | §4.4 (RQ3) | bootstrap confidence intervals for the order-of-mention tau, and KernelSHAP importances for the structured SVM |
| `Sec4-5_temporal_performance.csv` | 10 | §4.5 | temporal-split performance (train < 2021, test ≥ 2021) against the cross-validation figures |
| `Sec4-5_temporal_ablation.csv` | 5 | §4.5, §5.4 | temporal-split feature ablations |
