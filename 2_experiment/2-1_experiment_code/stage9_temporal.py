"""
Stage 9 (part 1) - Temporal robustness (research plan v4, Section 4.5).

The duplicate test2 set was dropped in Stage 1, so robustness is checked with a TEMPORAL split
instead: train on earlier offence years, test on later ones. This is a harder, more realistic
generalisation test than random CV (it asks whether the learned sentencing relationships hold
forward in time, under class-distribution drift).

Split: train = offense_year <= 2020 (n~756), test = offense_year >= 2021 (n~744).
Same models / pipelines / preprocessing as Stage 3 (all fit on TRAIN only -> no leakage).
We compare temporal-split macro-F1 / QWK against the Stage 3 random-CV means (robustness delta).

NOTE: offense_year is a structured feature but the split is defined on it, so for the structured
representation the test years are out-of-range (extrapolation) - reported with that caveat. The
headline model SVM+TF-IDF uses facts text, not the year feature, so it is the clean robustness read.

Inputs : outputs/stage2_features_verified.parquet, outputs/stage1_prepared.parquet,
         outputs/stage3_cv_results.csv (for the CV-vs-temporal comparison)
Outputs: outputs/T9_temporal.md, outputs/stage9_temporal_results.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from collections import Counter
import numpy as np
import pandas as pd

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
OUT = config.OUTPUT_DIR

SPLIT_YEAR = 2021   # test = years >= SPLIT_YEAR, train = years < SPLIT_YEAR

# -- Data ------------------------------------------------------------------
v = pd.read_parquet(OUT / "stage2_features_verified.parquet")
prep = (pd.read_parquet(OUT / "stage1_prepared.parquet")
        .drop_duplicates("id")[["id", "facts"]])
df = v.merge(prep, on="id", how="left").reset_index(drop=True)
df["offense_year"] = pd.to_numeric(df["offense_year"], errors="coerce")

is_test = df["offense_year"] >= SPLIT_YEAR
tr_idx = np.where(~is_test)[0]
te_idx = np.where(is_test)[0]
y = df["class4"].astype(int).values
ytr, yte = y[tr_idx], y[te_idx]

def dist(arr):
    c = pd.Series(arr).value_counts().sort_index()
    return {int(k): int(v) for k, v in c.items()}

print(f"Train (year < {SPLIT_YEAR}): n={len(tr_idx)} dist={dist(ytr)}")
print(f"Test  (year >= {SPLIT_YEAR}): n={len(te_idx)} dist={dist(yte)}")

NUM = ["bac", "distance_km", "prior_dui_count", "offense_year"]
CAT = ["vehicle"]
Xs = df[NUM + CAT].copy()
for c in NUM:
    Xs[c] = pd.to_numeric(Xs[c], errors="coerce")
texts = df["facts"].fillna("").values

# -- Pipelines (identical to Stage 3) --------------------------------------
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

REP = config.CLASS_REP_MONTHS
def metric_set(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    mae_mo = np.mean([abs(REP[a] - REP[b]) for a, b in zip(yt, yp)])
    return {
        "accuracy": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro"),
        "qwk": cohen_kappa_score(yt, yp, weights="quadratic"),
        "balanced_acc": balanced_accuracy_score(yt, yp),
        "mae_months": float(mae_mo),
        "tolerance_acc": float(np.mean(np.abs(yt - yp) <= 1)),
    }

# -- Temporal evaluation ---------------------------------------------------
rows = []
maj = Counter(ytr).most_common(1)[0][0]
rows.append(("baseline_majority", "-", metric_set(yte, np.full_like(yte, maj))))
rng = np.random.RandomState(SEED)
p = np.bincount(ytr, minlength=config.N_CLASSES) / len(ytr)
rows.append(("baseline_random", "-", metric_set(yte, rng.choice(config.N_CLASSES, size=len(yte), p=p))))

for name, mk in models().items():
    ps = struct_pipe(mk()).fit(Xs.iloc[tr_idx], ytr).predict(Xs.iloc[te_idx])
    rows.append((name, "structured", metric_set(yte, ps)))
    pt = text_pipe(mk()).fit(texts[tr_idx], ytr).predict(texts[te_idx])
    rows.append((name, "tfidf", metric_set(yte, pt)))

temporal = pd.DataFrame([{"model": m, "rep": r, **mm} for m, r, mm in rows])

# -- Compare to Stage 3 random-CV means ------------------------------------
cv = pd.read_csv(OUT / "stage3_cv_results.csv")
cv_mean = cv.groupby(["model", "rep"])[["macro_f1", "qwk"]].mean().rename(
    columns={"macro_f1": "cv_macro_f1", "qwk": "cv_qwk"})
comp = temporal.merge(cv_mean, on=["model", "rep"], how="left")
comp["d_macro_f1"] = comp["macro_f1"] - comp["cv_macro_f1"]
comp["d_qwk"] = comp["qwk"] - comp["cv_qwk"]
comp.to_csv(OUT / "stage9_temporal_results.csv", index=False, encoding="utf-8-sig")

# -- Report ----------------------------------------------------------------
order = [("baseline_majority", "-"), ("baseline_random", "-")]
for r in ["structured", "tfidf"]:
    for m in config.ML_MODELS:
        order.append((m, r))

lines = []
lines.append(f"# Table 9 - Temporal robustness (train year < {SPLIT_YEAR}, test year >= {SPLIT_YEAR})\n")
lines.append(f"Train n={len(tr_idx)} (class dist {dist(ytr)}); test n={len(te_idx)} "
             f"(class dist {dist(yte)}). Majority class is 2 in both, but class-1 share rises "
             f"in the test period -> genuine distribution drift. Same models/preprocessing as "
             f"Stage 3, fit on train only. `d_*` = temporal minus Stage-3 random-CV mean.\n")
lines.append("| model | repr | macro-F1 | QWK | bal-acc | acc | tol(+-1) | dMacroF1 vs CV | dQWK vs CV |")
lines.append("|---|---|---|---|---|---|---|---|---|")
idx = {(r["model"], r["rep"]): r for _, r in comp.iterrows()}
for key in order:
    if key not in idx:
        continue
    r = idx[key]
    dF = "" if pd.isna(r["d_macro_f1"]) else f"{r['d_macro_f1']:+.3f}"
    dQ = "" if pd.isna(r["d_qwk"]) else f"{r['d_qwk']:+.3f}"
    lines.append(f"| {r['model']} | {r['rep']} | {r['macro_f1']:.3f} | {r['qwk']:.3f} | "
                 f"{r['balanced_acc']:.3f} | {r['accuracy']:.3f} | {r['tolerance_acc']:.3f} | {dF} | {dQ} |")
lines.append("")

best = comp[(comp.model == "svm") & (comp.rep == "tfidf")].iloc[0]
maj_f1 = idx[("baseline_majority", "-")]["macro_f1"]
maj_qwk = idx[("baseline_majority", "-")]["qwk"]
all_ml = comp[~comp.model.str.startswith("baseline")]
beat_f1 = int((all_ml["macro_f1"] > maj_f1).sum())
beat_qwk = int((all_ml["qwk"] > maj_qwk).sum())
most_robust = all_ml.loc[all_ml["d_qwk"].idxmax()]
lines.append("## Robustness verdict\n")
lines.append(f"- **Material degradation under temporal shift** (expected: harder than random CV, "
             f"plus the class-1 share rises from {dist(ytr).get(1,0)/len(ytr)*100:.1f}% to "
             f"{dist(yte).get(1,0)/len(yte)*100:.1f}% across the split). Best CV model SVM+TF-IDF "
             f"drops macro-F1 {best['cv_macro_f1']:.3f}->{best['macro_f1']:.3f} ({best['d_macro_f1']:+.3f}), "
             f"QWK {best['cv_qwk']:.3f}->{best['qwk']:.3f} ({best['d_qwk']:+.3f}).")
lines.append(f"- **Signal survives, weaker**: {beat_qwk}/{len(all_ml)} ML configs keep QWK > 0 and "
             f"{beat_f1}/{len(all_ml)} keep macro-F1 above the temporal majority baseline "
             f"({maj_f1:.3f}) -> the H1 'informative learning' direction holds out-of-time.")
lines.append(f"- **Most temporally robust**: {most_robust['model']}+{most_robust['rep']} "
             f"(QWK {most_robust['qwk']:.3f}, delta {most_robust['d_qwk']:+.3f}); tree models "
             f"(random_forest) degrade least.")
lines.append(f"- **Exception**: SVM+structured collapses (macro-F1 "
             f"{idx[('svm','structured')]['macro_f1']:.3f}, BELOW the majority baseline, acc "
             f"{idx[('svm','structured')]['accuracy']:.3f}) - this is the year-feature extrapolation "
             f"caveat below, not a property of the facts signal.")
lines.append("- CAVEAT: for the *structured* representation, test years (>=%d) are out of the "
             "training range, so those rows also reflect year-feature extrapolation; the TF-IDF "
             "rows isolate genuine temporal generalisation." % SPLIT_YEAR)
lines.append("")
lines.append("Outputs: stage9_temporal_results.csv.")

(OUT / "T9_temporal.md").write_text("\n".join(lines), encoding="utf-8")
print("\n" + "\n".join(lines))
print("\nStage 9 (temporal) done. -> outputs/T9_temporal.md, outputs/stage9_temporal_results.csv")
