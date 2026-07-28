# -*- coding: utf-8 -*-
"""Stage 9b - Temporal-split factor ablations (RQ for the 2021 legal-regime change).
On the STRUCTURED representation (the only one with per-factor inputs), refit the best structured
model on the temporal split (train year < 2021, test >= 2021) with (a) all five factors,
(b) prior-DUI count dropped, (c) offence year dropped, and report temporal QWK each. Also compare
the TreeSHAP importance of prior-DUI count in the pre-2021 vs post-2021 subsets.

Inputs : outputs/stage2_features_verified.parquet
Outputs: outputs/stage9b_ablation.csv
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
import lightgbm as lgb
import shap
import config
config.set_seed()
SEED = config.SEED
OUT = config.OUTPUT_DIR
SPLIT_YEAR = 2021
LABELS = [0, 1, 2, 3]

v = pd.read_parquet(OUT / "stage2_features_verified.parquet")
v["offense_year"] = pd.to_numeric(v["offense_year"], errors="coerce")
NUM_ALL = ["bac", "distance_km", "prior_dui_count", "offense_year"]
CAT = ["vehicle"]
for c in NUM_ALL:
    v[c] = pd.to_numeric(v[c], errors="coerce")
y = v["class4"].astype(int).values
is_test = v["offense_year"] >= SPLIT_YEAR
tr, te = np.where(~is_test)[0], np.where(is_test)[0]
ytr, yte = y[tr], y[te]

def struct_logreg(num):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced",
                     max_iter=2000, random_state=SEED))])

def temporal_qwk(num):
    X = v[num + CAT]
    p = struct_logreg(num).fit(X.iloc[tr], ytr).predict(X.iloc[te])
    return (cohen_kappa_score(yte, p, labels=LABELS, weights="quadratic"),
            f1_score(yte, p, average="macro"))

variants = [("all five factors", NUM_ALL),
            ("drop prior-DUI count", [c for c in NUM_ALL if c != "prior_dui_count"]),
            ("drop offence year", [c for c in NUM_ALL if c != "offense_year"])]
rows = []
print("=== Temporal ablation (structured logistic regression; best structured model) ===")
for name, num in variants:
    q, f1 = temporal_qwk(num)
    rows.append(dict(analysis="temporal_ablation", variant=name, qwk=round(q, 3), macro_f1=round(f1, 3)))
    print(f"  {name:24s} QWK={q:.3f}  macro-F1={f1:.3f}")

# --- prior-DUI count SHAP importance: pre-2021 vs post-2021 subsets (LightGBM structured) ---
print("\n=== TreeSHAP importance of prior-DUI count, pre- vs post-2021 ===")
def shap_prior_importance(idx):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_ALL),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    Xt = pre.fit_transform(v[NUM_ALL + CAT].iloc[idx])
    names = NUM_ALL + list(pre.named_transformers_["cat"].get_feature_names_out(CAT))
    m = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                           random_state=SEED, verbose=-1).fit(Xt, y[idx])
    raw = shap.TreeExplainer(m).shap_values(Xt)
    sv = np.stack(raw, axis=2) if isinstance(raw, list) else np.asarray(raw)
    col_imp = np.abs(sv).mean(axis=0).mean(axis=1)
    return dict(zip(names, col_imp))

imp_pre = shap_prior_importance(tr)
imp_post = shap_prior_importance(te)
pp, qq = imp_pre["prior_dui_count"], imp_post["prior_dui_count"]
rows.append(dict(analysis="prior_shap", variant="pre-2021", qwk=round(pp, 3), macro_f1=""))
rows.append(dict(analysis="prior_shap", variant="post-2021", qwk=round(qq, 3), macro_f1=""))
print(f"  prior-DUI mean|SHAP|  pre-2021 = {pp:.3f}   post-2021 = {qq:.3f}   "
      f"({'lower' if qq < pp else 'higher'} post-2021)")

pd.DataFrame(rows).to_csv(OUT / "stage9b_ablation.csv", index=False, encoding="utf-8-sig")
print("\nStage 9b done -> outputs/stage9b_ablation.csv")
