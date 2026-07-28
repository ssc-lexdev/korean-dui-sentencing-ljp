"""
config.py - Shared settings and constants for the Korean drunk-driving LJP study
(research plan v4). Stage 0 deliverable. Every experiment script imports this
module to share paths, the random seed, and project-wide constants.
"""
from __future__ import annotations
import os
import random
import pathlib

# -- Paths -----------------------------------------------------------------
CODE_DIR    = pathlib.Path(__file__).resolve().parent          # 2_experiment/2-1_experiment_code
BASE_DIR    = CODE_DIR.parent.parent                            # repository root
DATASET_DIR = BASE_DIR / "dataset"
EXP_DIR     = BASE_DIR / "experiment_v4"
OUTPUT_DIR  = EXP_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "valid", "test", "test2"]

# -- Reproducibility -------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Fix the global seed (numpy, random). Stage 6 calls both LLMs over HTTP APIs,
    so no local model seeding is involved."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

# -- Target: 4-class ordinal relabeling (plan v4, Section 1) ---------------
# Original imprisonment_with_labor_lv (0..4) -> 4 classes (merge lv1 & lv2)
#   0: fine / no imprisonment   1: short term (3-10 mo)
#   2: medium term (12-17 mo)   3: long term (18-30 mo)
LV_TO_CLASS = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
CLASS_NAMES = {0: "fine/none", 1: "short", 2: "medium", 3: "long"}
N_CLASSES = 4
TARGET_FIELD = "imprisonment_with_labor_lv"
TARGET_CASENAME = "도로교통법위반(음주운전)"   # LBOX casename literal (Road Traffic Act, DUI)

# Representative months per class (for the month-level MAE reported alongside QWK, v4 Section 5)
CLASS_REP_MONTHS = {0: 0, 1: 7, 2: 13, 3: 21}

# -- ML-A structured sentencing factors (v4 Section 4.1; Hwang & Eom 2022 four fields + year)
FEATURES_A = ["bac", "distance_km", "vehicle", "prior_dui_count", "offense_year"]
FEATURES_A_EN = {
    "bac": "blood alcohol concentration",
    "distance_km": "driving distance (km)",
    "vehicle": "vehicle type",
    "prior_dui_count": "number of prior DUI convictions",
    "offense_year": "year of offense",
}
# Feature excluded from the temporal-split main analysis (v4 Section 8)
FEATURE_EXCLUDE_TEMPORAL = ["offense_year"]

# -- ML models (2 tree + 2 non-tree, v4 Section 4.1) ----------------------
ML_MODELS = ["svm", "logreg", "random_forest", "lightgbm"]
TREE_MODELS = ["random_forest", "lightgbm"]      # explained with TreeSHAP
NONTREE_MODELS = ["svm", "logreg"]               # linear -> coefficients / RBF-SVM -> KernelSHAP, LIME

# -- Evaluation (v4 Sections 5 & 6) ---------------------------------------
PRIMARY_METRICS = ["macro_f1", "qwk"]
AUX_METRICS = ["balanced_acc", "mae_level", "mae_months", "tolerance_acc"]
CV_N_SPLITS = 5
CV_N_REPEATS = 5
MULTIPLE_COMPARISON_CORRECTION = "holm"   # Holm / BH

# -- LLMs (v4 Section 4.2) ------------------------------------------------
# The identifiers actually called in Stage 6. gpt-5.5 was planned but was not
# available on the key, so gpt-5 was used; the open model was called through
# the OpenRouter API, not run locally.
LLM_CLOSED = "gpt-5"                       # OpenAI API (Stage 6)
LLM_OPEN = "qwen/qwen3-8b"                 # OpenRouter API (Stage 6)
LLM_TEMPERATURE = 0.0
LLM_VARIANTS = ["zero_shot_closed_book", "few_shot_open_book"]
FEW_SHOT_K = N_CLASSES                     # one worked example per class

# Statute and sentencing-guideline text used as the open-book context. The
# original run assembled this from two .docx files; the exact resulting string
# is shipped here so stage6 reproduces the prompt without them.
LAW_CONTEXT = CODE_DIR / "law_context.txt"

# -- Cross-model explanation agreement (v4 Section 7) ---------------------
FACTOR_SPACE = ["bac", "distance_km", "vehicle", "prior_dui_count", "offense_year"]
AGREEMENT_METRICS = ["kendall_tau", "jaccard_at_k"]
AGREEMENT_PERMUTATION_N = 10000

# -- Temporal split (v4 Section 8) ----------------------------------------
TEMPORAL_TRAIN_MAX_YEAR = 2020   # train <= 2020
TEMPORAL_TEST_YEAR = 2021        # test == 2021


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    set_seed()
    print("config loaded OK")
    print(f"  BASE_DIR    = {BASE_DIR}")
    print(f"  DATASET_DIR = {DATASET_DIR}  (exists: {DATASET_DIR.exists()})")
    print(f"  OUTPUT_DIR  = {OUTPUT_DIR}")
    print(f"  SEED        = {SEED}")
    print(f"  4-class map = {LV_TO_CLASS}")
    print(f"  ML-A feats  = {FEATURES_A}")
    print(f"  ML models   = {ML_MODELS}")
    print(f"  LLMs        = closed:{LLM_CLOSED} / open:{LLM_OPEN}")
    print(f"  law context = {LAW_CONTEXT}  (exists: {LAW_CONTEXT.exists()})")
