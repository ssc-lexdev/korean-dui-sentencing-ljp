# Computing Environment and Reproducibility (Stage 0)

This document records the computing environment for the experiments, written so it
can be cited directly in the paper's *Reproducibility* / *Experimental Setup* section.

## Hardware

| Component | Specification |
|---|---|
| Workstation | Windows 11 Home (build 26200) |
| CPU | local workstation (CPU-only for ML, statistics, and explanation) |
| GPU | Intel Iris Xe Graphics (integrated, ~1 GB) — **no CUDA / no NVIDIA GPU, and none required** |

**No GPU is needed to reproduce this study.** Both LLMs are reached over HTTP APIs rather
than run locally, so no model weights are downloaded and no CUDA device is used: the
proprietary model through the OpenAI API, and the open-weight model (Qwen3-8B-Instruct)
through the OpenRouter API. Every other stage — data preparation, classical ML, SHAP,
statistical testing, agreement analysis, temporal robustness — runs on the local CPU.

An earlier plan was to run Qwen3-8B locally on a Google Colab GPU. That is not what was
done, and Colab was not used for any reported result.

## Software

| Component | Version |
|---|---|
| Python | 3.13.9 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scipy | 1.16.3 |
| scikit-learn | 1.7.2 |
| lightgbm | 4.6.0 |
| xgboost | 3.2.0 |
| shap | 0.52.0 |
| lime | (latest at install time) |
| statsmodels | 0.14.5 |
| matplotlib | 3.10.6 |
| openai | 2.36.0 |
| datasets | 5.0.0 |
| pyarrow | 21.0.0 |

Exact pins are listed in `requirements.txt`. No deep-learning stack is needed: both LLMs are
called over HTTP, so `torch`, `transformers`, and `accelerate` are not required anywhere.

## Reproducibility settings

- **Global seed**: `SEED = 42`, fixed via `config.set_seed()` for `random` and `numpy`.
  No local model training is involved in Stage 6, so no framework seeding applies there.
- **Cross-validation**: repeated stratified, group-aware k-fold
  (`CV_N_SPLITS = 5`, `CV_N_REPEATS = 5`); the fixed LBOX split is also reported for
  comparability with prior work.
- **LLM determinism**: `temperature = 0.0` for the open-weight model, and the minimal
  reasoning-effort tier for the proprietary one; the model identifier, call timestamp, and
  the fixed few-shot exemplar set are logged for every run. Proprietary-API outputs may
  still drift across provider updates — an inherent limitation acknowledged in the paper.
  The released response caches make the reported numbers reproducible regardless.
- **Leakage control**: all preprocessing (TF-IDF vectorizer fitting, scaling, resampling)
  is fit **inside the training fold only**; the rule-based sentencing-factor extractor is
  **frozen on the training fold** before being applied elsewhere.

## Installation

```bash
# One CPU environment covers every stage, including the LLM stage.
python -m pip install -r requirements.txt
```

## Dataset note

The corpus is the drunk-driving (도로교통법위반(음주운전)) subset of the public
LBOX OPEN `ljp_criminal` benchmark: 1,644 records, 1,500 unique cases after removing the
`test2` duplicates. The subset **is** redistributed here, in `1_dataset/raw/`, under the
source corpus's CC BY-NC 4.0 licence; see the README for the attribution and the list of
modifications. The official source is the Hugging Face dataset `lbox/lbox_open`.

`2_experiment/2-1_experiment_code/prepare_workspace.py` stages `1_dataset/` into the working layout the stage scripts
expect; the staged copies under `dataset/` and `experiment_v4/` are git-ignored.
