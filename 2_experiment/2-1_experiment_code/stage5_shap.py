"""
Stage 5 - SHAP/LIME explanation of the structured model (research plan v4, Section 7, H3).

Explains a LightGBM model on the verified 5 sentencing factors (ML-A). Produces TreeSHAP
global importance (bar + beeswarm), a local waterfall for one case, a LIME cross-check, and
the ML factor-importance ranking r_ML over the shared factor space (for Stage 8 agreement).
Tests the prior hypothesis (Hwang & Eom 2022): prior_dui_count and offense_year are top
contributors, bac comparatively weak.

Input : outputs/stage2_features_verified.parquet
Outputs: F1_shap_global_bar.png, F2_shap_beeswarm.png, F3_shap_waterfall.png,
         stage5_factor_ranking.csv (r_ML), T5_shap_summary.md
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Figure labels are English; use the default font (DejaVu Sans includes the Unicode minus
# U+2212) so SHAP waterfall negative-value labels render correctly instead of as a box.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import shap

import config
config.set_seed()
OUT = config.OUTPUT_DIR

df = pd.read_parquet(OUT / "stage2_features_verified.parquet")
y = df["class4"].astype(int).values
NUM = ["bac", "distance_km", "prior_dui_count", "offense_year"]
CAT = ["vehicle"]
X = df[NUM + CAT].copy()
for c in NUM:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# Preprocess (impute + one-hot); explanation is on the fitted model over the full set.
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
Xt = pre.fit_transform(X)
oh_names = list(pre.named_transformers_["cat"].get_feature_names_out(CAT))
feat_names = NUM + oh_names
Xt = pd.DataFrame(Xt, columns=feat_names)

model = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                           random_state=config.SEED, verbose=-1).fit(Xt, y)
print(f"Model fitted on {len(Xt)} cases, {len(feat_names)} columns, {config.N_CLASSES} classes")

# -- TreeSHAP --------------------------------------------------------------
explainer = shap.TreeExplainer(model)
raw = explainer.shap_values(Xt)
# Normalize to (n_samples, n_features, n_classes)
if isinstance(raw, list):
    sv = np.stack(raw, axis=2)
else:
    arr = np.asarray(raw)
    n_s, n_f, n_c = len(Xt), len(feat_names), config.N_CLASSES
    sv = arr if arr.shape == (n_s, n_f, n_c) else arr.transpose(
        {(n_c, n_s, n_f): (1, 2, 0), (n_c, n_f, n_s): (2, 1, 0)}[arr.shape])
print(f"SHAP values shape: {sv.shape}")

# Per-column mean|SHAP| (avg over samples and classes), then aggregate one-hot vehicle cols.
col_imp = np.abs(sv).mean(axis=0).mean(axis=1)   # (n_features,)
imp = {}
for name, val in zip(feat_names, col_imp):
    base = "vehicle" if name.startswith("vehicle_") else name
    imp[base] = imp.get(base, 0.0) + val
rank = pd.Series(imp).sort_values(ascending=False)
print("\nFactor importance (mean|SHAP|):")
for k, v in rank.items():
    print(f"  {k:16s} {v:.4f}")

# r_ML ranking over the shared factor space (Stage 8)
rml = rank.reindex(config.FACTOR_SPACE).fillna(0.0)
rml_df = pd.DataFrame({"factor": rml.index, "shap_importance": rml.values,
                       "rank": rml.rank(ascending=False).astype(int).values})
rml_df.to_csv(OUT / "stage5_factor_ranking.csv", index=False, encoding="utf-8-sig")

# -- Plot 1: global bar ----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
r = rank.sort_values()
ax.barh([config.FEATURES_A_EN.get(k, k) for k in r.index], r.values, color="#2196F3")
ax.set_xlabel("mean(|SHAP value|)  (averaged over classes)")
ax.set_title("Global factor importance (LightGBM, TreeSHAP)")
plt.tight_layout()
fig.savefig(OUT / "F1_shap_global_bar.png", dpi=150)
# This panel is Figure 3 of the paper; write it where the released copy lives.
(config.BASE_DIR / "4_figures").mkdir(parents=True, exist_ok=True)
fig.savefig(config.BASE_DIR / "4_figures" / "Figure_3.png", dpi=150)
plt.close(fig)

# -- Plot 2: beeswarm for the long-imprisonment class (class 3) ------------
TARGET = 3
expl = shap.Explanation(values=sv[:, :, TARGET], data=Xt.values, feature_names=feat_names)
plt.figure(figsize=(7, 4))
shap.plots.beeswarm(expl, max_display=len(feat_names), show=False)
plt.title("SHAP beeswarm - class 3 (long imprisonment)")
plt.tight_layout(); plt.savefig(OUT / "F2_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

# -- Plot 3: local waterfall for one high-class case -----------------------
idx = int(np.where(y == 3)[0][0])
ev = explainer.expected_value
base_val = ev[TARGET] if np.ndim(ev) else ev
one = shap.Explanation(values=sv[idx, :, TARGET], base_values=base_val,
                       data=Xt.iloc[idx].values, feature_names=feat_names)
plt.figure(figsize=(7, 4))
shap.plots.waterfall(one, show=False)
plt.title(f"SHAP waterfall - case id {int(df['id'].iloc[idx])} (true class {y[idx]})")
plt.tight_layout(); plt.savefig(OUT / "F3_shap_waterfall.png", dpi=150, bbox_inches="tight"); plt.close()

# -- LIME cross-check on one case -----------------------------------------
lime_note = ""
try:
    from lime.lime_tabular import LimeTabularExplainer
    le = LimeTabularExplainer(Xt.values, feature_names=feat_names,
                              class_names=[str(c) for c in range(config.N_CLASSES)],
                              discretize_continuous=True, random_state=config.SEED)
    lex = le.explain_instance(Xt.iloc[idx].values, model.predict_proba,
                              num_features=len(feat_names), labels=(TARGET,))
    lime_top = [f.split(" ")[0] if "vehicle" not in f else "vehicle"
                for f, _ in sorted(lex.as_list(label=TARGET), key=lambda t: -abs(t[1]))][:3]
    lime_note = f"LIME top-3 factors for case {int(df['id'].iloc[idx])}: {lime_top}."
except Exception as e:
    lime_note = f"LIME cross-check skipped ({e})."
print("\n" + lime_note)

# -- Summary ---------------------------------------------------------------
top = list(rank.index[:2])
hyp_ok = ("prior_dui_count" in top or "offense_year" in top) and rank.get("bac", 0) < rank.iloc[0]
lines = ["# Table 5 / Figures - SHAP factor importance (H3)\n",
         "LightGBM on the 5 verified sentencing factors; TreeSHAP mean|SHAP| over classes.\n",
         "| factor | mean\\|SHAP\\| | rank |", "|---|---|---|"]
for i, (k, v) in enumerate(rank.items(), 1):
    lines.append(f"| {config.FEATURES_A_EN.get(k,k)} | {v:.4f} | {i} |")
lines.append(f"\n**Prior hypothesis (Hwang & Eom 2022)**: prior_dui_count / offense_year are top "
             f"contributors, bac comparatively weak. Observed top-2 = {top}; "
             f"hypothesis {'SUPPORTED' if hyp_ok else 'NOT clearly supported'}.\n")
lines.append(lime_note + "\n")
lines.append("Figures: F1_shap_global_bar.png, F2_shap_beeswarm.png, F3_shap_waterfall.png.\n")
lines.append("r_ML factor ranking saved to stage5_factor_ranking.csv (for Stage 8 cross-model agreement).")
(OUT / "T5_shap_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("\nStage 5 done.")
