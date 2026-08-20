# Supplementary Material

> Companion to `draft.md`. Collects the diagnostic and full-detail material deliberately kept out of
> the main text to keep it focused. All numbers are from the verified Stage 1–9 outputs; each section
> names its source file under `docs/results/` and `outputs/`.

## S1. Feature extraction and expert verification (supports §3.1)

All 1,500 unique cases were reviewed case-by-case by a practising attorney; only incorrect values
were corrected, yielding the ground-truth feature table used throughout.

| Factor | Corrections | Extraction accuracy |
|---|---|---|
| Blood-alcohol concentration (BAC) | 7 | 99.5% |
| Driving distance | 11 | 99.3% |
| Vehicle type | 92 | 93.9% |
| Prior-DUI/refusal count | 4 | 99.7% |
| Offence year | 1 | 99.9% |
| **All values** | **114 / 7,500** | **98.5%** |

Most vehicle corrections were model names the rule had left as "other"; nine cases remain "other"
because the judgment anonymises the vehicle. Remaining missing values (BAC 3, distance 37) are
genuine and imputed inside the training fold. Source: `docs/results/stage2_verification.md`.

## S2. Full ML performance, including secondary metrics (supports §4.1, Table 2)

Repeated stratified group-aware 5×5 CV; mean (std) over 25 folds. Extends main Table 2 with mean
absolute error in months. Source: `2_experiment/2-2_experiment_result/results/Sec4-1_Table2_ml_performance_per_fold.csv`.

| Model | Repr. | macro-F1 | QWK | Bal-acc | Acc | MAE (months) | Tol(±1) |
|---|---|---|---|---|---|---|---|
| Majority baseline | – | 0.202 | 0.000 | 0.250 | 0.678 | 3.15 | 0.857 |
| Random baseline | – | 0.253 | −0.002 | 0.254 | 0.498 | 5.28 | 0.757 |
| SVM | structured | 0.345 | 0.269 | 0.492 | 0.348 | 6.53 | 0.748 |
| Logistic Reg. | structured | 0.319 | 0.240 | 0.476 | 0.340 | 6.38 | 0.771 |
| Random Forest | structured | 0.390 | 0.291 | 0.416 | 0.605 | 3.79 | 0.853 |
| LightGBM | structured | 0.375 | 0.233 | 0.371 | 0.595 | 4.11 | 0.826 |
| SVM | TF-IDF | 0.451 | 0.348 | 0.489 | 0.606 | 3.83 | 0.848 |
| Logistic Reg. | TF-IDF | 0.432 | 0.328 | 0.524 | 0.517 | 4.70 | 0.825 |
| Random Forest | TF-IDF | 0.378 | 0.290 | 0.385 | 0.684 | 3.00 | 0.880 |
| LightGBM | TF-IDF | 0.382 | 0.295 | 0.373 | 0.682 | 3.04 | 0.881 |

## S3. LLM prediction distributions and confusion (supports §4.2–4.3)

Parsing succeeded for 100% of responses in all four configurations (no refusals or format failures).
Predicted-class shares against the true distribution (0/1/2/3 = 14/7/68/11%):

| Config | pred 0 | pred 1 | pred 2 | pred 3 |
|---|---|---|---|---|
| SVM + TF-IDF (best ML) | 0.15 | 0.14 | 0.61 | 0.10 |
| GPT-5, zero-shot | 0.16 | **0.69** | 0.14 | 0.01 |
| Qwen3-8B, zero-shot | 0.00 | 0.09 | **0.80** | 0.11 |

Confusion matrices (true × predicted counts) for the two zero-shot LLMs make the opposite collapse
explicit (also Figure 2):

GPT-5 zero-shot — predictions pile into column 1:
```
pred:    0    1    2   3
true 0  69  136   10   0
true 1  47   57    2   1
true 2 119  742  143  13
true 3   5  100   53   3
```
Qwen3-8B zero-shot — predictions pile into column 2:
```
pred:   0   1    2    3
true 0  0  42  163   10
true 1  3  30   71    3
true 2  1  55  845  116
true 3  0   3  125   33
```
Source: `docs/results/stage6_distribution_check.md`.

## S4. Ordinal (±1) McNemar — full table (supports §4.2)

Main Table 3 reports exact-match McNemar; under the ordinal definition (|predicted − true| ≤ 1) the
LLMs are marginally ahead, because their errors are near-misses. Best ML = SVM + TF-IDF.
*b* = ML within-band / LLM not; *c* = ML not / LLM within-band. Source: `outputs/T7_h2_tests.md`.

| LLM config | ML within±1 | LLM within±1 | *b* | *c* | *p* (Holm) | Favours |
|---|---|---|---|---|---|---|
| GPT-5, zero-shot | 0.850 | 0.843 | 187 | 177 | 0.64 | tie |
| Qwen3-8B, zero-shot | 0.850 | 0.880 | 79 | 124 | 0.004 | LLM |
| GPT-5, few-shot | 0.850 | 0.903 | 68 | 147 | <10⁻⁴ | LLM |
| Qwen3-8B, few-shot | 0.850 | 0.888 | 63 | 120 | 0.0001 | LLM |

## S5. LLM factor fidelity — perception vs judgement (supports §4.3)

Numeric claims inside each LLM rationale, parsed and compared to the verified features. "Parsed"
counts cases where the model stated a value (coverage differs by verbosity). Source:
`docs/results/stage6_factor_fidelity.md`.

| Config | BAC parsed | BAC match (±0.005) | Prior parsed | Prior exact |
|---|---|---|---|---|
| GPT-5, zero-shot | 1,497/1,500 | 99.5% | 490/1,500 | 95.1% |
| GPT-5, few-shot | 1,497/1,500 | 99.5% | 1,050/1,500 | 97.3% |
| Qwen3-8B, zero-shot | 347/1,500 | 98.8% | 1/1,500 | – |
| Qwen3-8B, few-shot | 1,497/1,500 | 99.7% | 1,354/1,500 | 95.2% |

GPT-5's qualitative BAC descriptors track the true value monotonically (mean actual BAC): low 0.062
(n=155), medium 0.102 (22), high 0.156 (37), very-high 0.156, max 0.306 (168). Residual BAC
mismatches are largely multi-incident cases — e.g. id 4675 records two same-day readings (0.211 at
21:49, 0.248 at 22:50 after re-drinking); the rule retains the higher (0.248) while GPT stated the
first (0.211). Both LLMs thus read facts reliably while mispredicting the sentence.

## S6. Explanation-agreement signals (supports §4.4, Table 4)

Two signals derive *r_LLM* from the cited factors: citation **frequency** and **order** of first
mention. Frequency saturates when a model lists every factor (GPT cites all five at ~99%), so it is
uninformative there; order is the discriminative signal used in the main text. Source:
`2_experiment/2-2_experiment_result/results/Sec4-4_Table4_explanation_agreement.csv`.

Frequency-signal agreement vs *r_ML* (for contrast with main Table 4's order signal):

| Config | Kendall τ (freq) | perm *p* | Jaccard@2 | Jaccard@3 |
|---|---|---|---|---|
| GPT-5, zero-shot | −0.20 | 0.82 | 0.33 | 0.50 |
| Qwen3-8B, zero-shot | 0.80 | 0.083 | 1.00 | 1.00 |
| GPT-5, few-shot | 0.60 | 0.23 | 0.33 | 0.50 |
| Qwen3-8B, few-shot | 0.00 | 1.00 | 0.33 | 0.50 |

Factor citation frequency (% of cases), showing GPT's saturation vs Qwen zero-shot's selectivity:

| Config | prior | BAC | distance | year | vehicle |
|---|---|---|---|---|---|
| GPT-5, zero-shot | 96.3 | 99.3 | 99.1 | 99.7 | 98.1 |
| Qwen3-8B, zero-shot | 75.4 | 99.8 | 51.0 | 32.9 | 26.9 |
| GPT-5, few-shot | 100.0 | 99.8 | 99.7 | 100.0 | 57.1 |
| Qwen3-8B, few-shot | 90.3 | 99.8 | 98.8 | 99.9 | 0.3 |

The Korean→canonical keyword map and the exact permutation procedure (5! = 120 orderings) are in
`stage8_agreement.py`.

## S7. Temporal robustness — full table (supports §4.5)

Train offence year < 2021 (n = 756) / test ≥ 2021 (n = 744); same models/pipelines, fit on train
only. `Δ` columns are temporal minus the Stage-3 random-CV mean. Source: `outputs/T9_temporal.md`,
`2_experiment/2-2_experiment_result/results/Sec4-5_temporal_performance.csv`.

| Model | Repr. | macro-F1 | QWK | Bal-acc | Acc | Tol(±1) | ΔmacroF1 | ΔQWK |
|---|---|---|---|---|---|---|---|---|
| Majority baseline | – | 0.203 | 0.000 | 0.250 | 0.684 | 0.899 | – | – |
| Random baseline | – | 0.264 | 0.012 | 0.269 | 0.500 | 0.742 | – | – |
| SVM | structured | 0.164 | 0.059 | 0.323 | 0.159 | 0.312 | −0.181 | −0.210 |
| Logistic Reg. | structured | 0.298 | 0.281 | 0.305 | 0.622 | 0.905 | −0.021 | +0.041 |
| Random Forest | structured | 0.280 | 0.279 | 0.331 | 0.505 | 0.750 | −0.110 | −0.012 |
| LightGBM | structured | 0.267 | 0.225 | 0.304 | 0.495 | 0.745 | −0.108 | −0.008 |
| SVM | TF-IDF | 0.257 | 0.197 | 0.348 | 0.387 | 0.590 | −0.193 | −0.150 |
| Logistic Reg. | TF-IDF | 0.324 | 0.216 | 0.419 | 0.374 | 0.699 | −0.108 | −0.112 |
| Random Forest | TF-IDF | 0.322 | 0.276 | 0.372 | 0.559 | 0.797 | −0.056 | −0.014 |
| LightGBM | TF-IDF | 0.291 | 0.212 | 0.313 | 0.566 | 0.790 | −0.091 | −0.083 |

The class-1 share rises from 3.8% (train) to 10.5% (test) — genuine label drift. All eight ML
configurations keep QWK > 0; seven of eight keep macro-F1 above the temporal majority baseline
(0.203). SVM+structured falls below it because the post-2020 offence years lie outside the training
range (feature extrapolation); the year-agnostic TF-IDF rows are the clean reading.

## S8. Robustness analyses added in revision (supports §4.2, §4.4, §4.5)

These analyses back the revised quantitative claims. Each is reproduced by a committed script;
results are written to `outputs/`.

**(a) Ordinal-aware paired comparison — bootstrap QWK difference** (`stage7b_qwk_bootstrap.py`).
Case-bootstrap 95% CI (B = 10,000 over the 1,500 shared cases, percentile) for the QWK difference
between the best ML model (SVM + TF-IDF, OOF) and each LLM.

| LLM config | QWK (ML) | QWK (LLM) | ΔQWK | 95% CI | verdict |
|---|---|---|---|---|---|
| GPT-5, zero-shot | 0.357 | 0.180 | +0.177 | [0.127, 0.226] | ML > LLM |
| Qwen3-8B, zero-shot | 0.357 | 0.184 | +0.174 | [0.116, 0.231] | ML > LLM |
| Qwen3-8B, few-shot | 0.357 | 0.240 | +0.118 | [0.062, 0.172] | ML > LLM |
| GPT-5, few-shot | 0.357 | 0.336 | +0.021 | [−0.029, 0.071] | **on par** (CI spans 0) |

**(b) Order-of-mention Kendall τ — case-bootstrap** (`stage8b_tau_kernelshap.py`). Both independent
zero-shot rankings are stable: Qwen τ = 1.00 (95% CI [1.00, 1.00]); GPT τ = 0.80 (95% CI [0.80, 1.00]).
Top-set overlap Jaccard@2 = Jaccard@3 = 1.00 for both.

**(c) Model-agnostic KernelSHAP on the structured SVM** (`stage8b_tau_kernelshap.py`). Aggregated
mean|SHAP| over the five factors: BAC 0.0198, prior-DUI count 0.0197 (top two, essentially tied),
driving distance 0.0120, offence year 0.0088, vehicle ≈ 0. The top-two factor set matches the
TreeSHAP/LightGBM ranking, so the ranking is not an artefact of the tree model.

**(d) Temporal-split factor ablation** (`stage9b_temporal_ablation.py`). Structured representation,
best structured model (logistic regression): temporal QWK = 0.281 with all five factors, **0.066**
without prior-DUI count, and 0.193 without offence year — prior-DUI count is by far the dominant
structured factor. TreeSHAP mean|SHAP| of prior-DUI count is 1.46 (pre-2021) vs 1.98 (post-2021);
the factor remains the dominant structured predictor across the 2021 regime change (the absolute
magnitudes are not directly comparable, as each subset's model has its own output scale).

**(e) Disposition-leakage audit** (`stage_disposition_audit.py`). Over all 1,500 `facts` texts, a
present-tense sentencing of the defendant (`피고인을 … 징역/금고/벌금 … 처한다/선고한다`, i.e.
"the defendant is hereby sentenced to imprisonment / confinement / a fine of …") matched **0**
cases, and the holding heading `주문` (*jumun*, lit. "main text" — the section of a Korean
judgment that states the sentence) matched **0**; sentencing verbs that do appear (74.3% of
texts) are prior-conviction recitations in the criminal-record section. No current disposition leaks
into the model inputs.

## Supplementary figures

Two SHAP figures are provided as Supplemental Files (in `5_supplement/`):

- **Figure S1** — Per-class SHAP beeswarm for the long-imprisonment class (class 3): higher BAC and
  more prior-DUI convictions push predictions toward class 3, consistent with *r_ML* (main Figure 3).
- **Figure S2** — Local SHAP waterfall for one class-3 case (id 4575), illustrating the per-case
  additive decomposition from E[f(X)] to f(x). Regenerated with the default font so the Unicode-minus
  glyph renders correctly.

## Prompts and statutory context (supports §3.3)

`5_supplement/few_shot_open_book_prompt.md` documents, in Korean and in English translation, the full
few-shot open-book prompt: the
instruction/system prompt, the statutory penalties (Road Traffic Act Article 148-2 and the Criminal
Act suspended-sentence and sentencing-condition provisions), the Sentencing Commission's DUI/
unlicensed-driving guideline table, and the deterministic construction of the four class-balanced
examples. The zero-shot closed-book configuration uses only the instruction prompt (no statute,
guideline, or examples). Sections 2 and 3 of that file are a condensed reading version; the statute
and guideline string the models actually received is `2_experiment/2-1_experiment_code/law_context.txt`, which
`stage6_llm.py` reads directly. Source: `stage6_llm.py`.
