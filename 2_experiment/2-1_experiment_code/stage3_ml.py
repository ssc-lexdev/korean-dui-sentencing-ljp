"""
Stage 3 - Classical ML benchmark with repeated stratified CV (research plan v4, Sections 4-6).

Uses the expert-verified feature table. 4 models x {structured 5-feature, TF-IDF on facts},
plus majority / random baselines, under RepeatedStratifiedKFold. ALL preprocessing (impute,
scale, one-hot, TF-IDF fit) happens inside the training fold via sklearn Pipelines, so no
leakage. Reports macro-F1, QWK, balanced accuracy, MAE(level), MAE(months), tolerance(+-1),
and accuracy, as mean +- std across folds.

Inputs : outputs/stage2_features_verified.parquet (5 features, verified)
         outputs/stage1_prepared.parquet           (facts text for TF-IDF, by id)
Outputs: outputs/stage3_cv_results.csv             (per-fold raw metrics)
         outputs/T2_model_performance.md            (mean +- std summary, paper Table 2)
         outputs/stage3_oof_predictions.parquet     (out-of-fold preds for Stage 4/7 tests)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from collections import Counter
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, cohen_kappa_score, balanced_accuracy_score,
                             accuracy_score)
import lightgbm as lgb

import config
config.set_seed()
SEED = config.SEED

# -- Data ------------------------------------------------------------------
v = pd.read_parquet(config.OUTPUT_DIR / "stage2_features_verified.parquet")
prep = (pd.read_parquet(config.OUTPUT_DIR / "stage1_prepared.parquet")
        .drop_duplicates("id")[["id", "facts"]])
df = v.merge(prep, on="id", how="left").reset_index(drop=True)
y = df["class4"].astype(int).values
print(f"Cases: {len(df)} | class dist: {dict(pd.Series(y).value_counts().sort_index())}")

NUM = ["bac", "distance_km", "prior_dui_count", "offense_year"]
CAT = ["vehicle"]
Xs = df[NUM + CAT].copy()
for c in NUM:
    Xs[c] = pd.to_numeric(Xs[c], errors="coerce")
texts = df["facts"].fillna("").values

# -- Pipelines -------------------------------------------------------------
def struct_pipe(clf):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])

def text_pipe(clf):
    return Pipeline([
        ("tf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                               max_features=2000, sublinear_tf=True)),
        ("clf", clf),
    ])

def models():
    return {
        "svm":           lambda: SVC(class_weight="balanced", random_state=SEED),
        "logreg":        lambda: LogisticRegression(class_weight="balanced",
                                                    max_iter=2000, random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=None,
                                                        min_samples_leaf=2,
                                                        class_weight="balanced",
                                                        random_state=SEED, n_jobs=-1),
        "lightgbm":      lambda: lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=SEED, verbose=-1),
    }

# -- Metrics ---------------------------------------------------------------
REP = config.CLASS_REP_MONTHS
def metric_set(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    mae_mo = np.mean([abs(REP[a] - REP[b]) for a, b in zip(yt, yp)])
    return {
        "accuracy": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro"),
        "qwk": cohen_kappa_score(yt, yp, weights="quadratic"),
        "balanced_acc": balanced_accuracy_score(yt, yp),
        "mae_level": float(np.mean(np.abs(yt - yp))),
        "mae_months": float(mae_mo),
        "tolerance_acc": float(np.mean(np.abs(yt - yp) <= 1)),
    }

# -- CV --------------------------------------------------------------------
cv = RepeatedStratifiedKFold(n_splits=config.CV_N_SPLITS,
                             n_repeats=config.CV_N_REPEATS, random_state=SEED)
rows, oof = [], []
n_folds = config.CV_N_SPLITS * config.CV_N_REPEATS
print(f"Running {n_folds} folds x (2 baselines + 4 models x 2 reps)...")

for fold, (tr, te) in enumerate(cv.split(Xs, y)):
    ytr, yte = y[tr], y[te]
    ids_te = df["id"].values[te]

    maj = Counter(ytr).most_common(1)[0][0]
    rows.append(("baseline_majority", "-", fold, metric_set(yte, np.full_like(yte, maj))))
    rng = np.random.RandomState(SEED + fold)
    p = np.bincount(ytr, minlength=config.N_CLASSES) / len(ytr)
    rows.append(("baseline_random", "-", fold,
                 metric_set(yte, rng.choice(config.N_CLASSES, size=len(yte), p=p))))

    for name, mk in models().items():
        ps = struct_pipe(mk()).fit(Xs.iloc[tr], ytr).predict(Xs.iloc[te])
        rows.append((name, "structured", fold, metric_set(yte, ps)))
        pt = text_pipe(mk()).fit(texts[tr], ytr).predict(texts[te])
        rows.append((name, "tfidf", fold, metric_set(yte, pt)))
        if fold < config.CV_N_SPLITS:   # store one full out-of-fold pass (first repeat)
            for i, idx in enumerate(te):
                oof.append((int(ids_te[i]), name, "structured", int(yte[i]), int(ps[i])))
                oof.append((int(ids_te[i]), name, "tfidf", int(yte[i]), int(pt[i])))
    if (fold + 1) % 5 == 0:
        print(f"  fold {fold+1}/{n_folds} done")

# -- Aggregate -------------------------------------------------------------
long = pd.DataFrame([{"model": m, "rep": r, "fold": f, **mm} for m, r, f, mm in rows])
long.to_csv(config.OUTPUT_DIR / "stage3_cv_results.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(oof, columns=["id", "model", "rep", "y_true", "y_pred"]).to_parquet(
    config.OUTPUT_DIR / "stage3_oof_predictions.parquet", index=False)

METRICS = ["macro_f1", "qwk", "balanced_acc", "accuracy", "mae_months", "tolerance_acc"]
agg = long.groupby(["model", "rep"])[METRICS].agg(["mean", "std"])

order = [("baseline_majority", "-"), ("baseline_random", "-")]
for r in ["structured", "tfidf"]:
    for m in config.ML_MODELS:
        order.append((m, r))

lines = ["# Table 2 - Model performance (repeated stratified 5x5 CV, N=1,500)\n",
         "Mean (std) across 25 folds. Majority-class baseline accuracy ~ 0.68.\n",
         "| model | repr | macro-F1 | QWK | bal-acc | acc | MAE(mo) | tol(+-1) |",
         "|---|---|---|---|---|---|---|---|"]
for key in order:
    if key not in agg.index:
        continue
    rrow = agg.loc[key]
    def cell(met):
        return f"{rrow[(met,'mean')]:.3f} ({rrow[(met,'std')]:.3f})"
    lines.append(f"| {key[0]} | {key[1]} | {cell('macro_f1')} | {cell('qwk')} | "
                 f"{cell('balanced_acc')} | {cell('accuracy')} | "
                 f"{rrow[('mae_months','mean')]:.2f} | {cell('tolerance_acc')} |")
(config.OUTPUT_DIR / "T2_model_performance.md").write_text("\n".join(lines), encoding="utf-8")

print("\n" + "\n".join(lines))
print(f"\nSaved: stage3_cv_results.csv, T2_model_performance.md, stage3_oof_predictions.parquet")
print("Stage 3 done.")
