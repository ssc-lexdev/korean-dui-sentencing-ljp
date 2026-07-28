# -*- coding: utf-8 -*-
"""Regenerate supplementary SHAP figures S1 (beeswarm) and S2 (waterfall).
Fixes the minus-glyph box (uses DejaVu Sans, which has U+2212) and removes embedded titles
(captions are supplied separately). Saves into 5_supplement/."""
import os, warnings
# Run from the script's own directory so all paths are repo-relative (portable).
BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Use the default font (has the Unicode minus U+2212) and ASCII minus on axes.
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
DEST = os.path.join(os.path.dirname(os.path.dirname(BASE)), "5_supplement")
os.makedirs(DEST, exist_ok=True)

df = pd.read_parquet(OUT / "stage2_features_verified.parquet")
y = df["class4"].astype(int).values
NUM = ["bac", "distance_km", "prior_dui_count", "offense_year"]
CAT = ["vehicle"]
X = df[NUM + CAT].copy()
for c in NUM:
    X[c] = pd.to_numeric(X[c], errors="coerce")
pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
Xt = pre.fit_transform(X)
oh = list(pre.named_transformers_["cat"].get_feature_names_out(CAT))
feat = NUM + oh
Xt = pd.DataFrame(Xt, columns=feat)

model = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                           random_state=config.SEED, verbose=-1).fit(Xt, y)
explainer = shap.TreeExplainer(model)
raw = explainer.shap_values(Xt)
if isinstance(raw, list):
    sv = np.stack(raw, axis=2)
else:
    arr = np.asarray(raw); n_s, n_f, n_c = len(Xt), len(feat), config.N_CLASSES
    sv = arr if arr.shape == (n_s, n_f, n_c) else arr.transpose(
        {(n_c, n_s, n_f): (1, 2, 0), (n_c, n_f, n_s): (2, 1, 0)}[arr.shape])

# Readable feature labels (numerics prettified; vehicle one-hots tidied)
pretty = {"bac": "blood alcohol concentration", "distance_km": "driving distance (km)",
          "prior_dui_count": "prior DUI convictions", "offense_year": "year of offence"}
def lab(n):
    if n in pretty: return pretty[n]
    if n.startswith("vehicle_"): return "vehicle: " + n.split("_", 1)[1]
    return n
labels = [lab(n) for n in feat]

TARGET = 3

# ---------- S1: beeswarm, class 3 (no embedded title) ----------
expl = shap.Explanation(values=sv[:, :, TARGET], data=Xt.values, feature_names=labels)
plt.figure(figsize=(7.5, 4.2))
shap.plots.beeswarm(expl, max_display=len(feat), show=False)
plt.xlabel("SHAP value (impact on class-3 prediction)")
plt.tight_layout()
plt.savefig(os.path.join(DEST, "Figure_S1.png"), dpi=200, bbox_inches="tight"); plt.close()

# ---------- S2: waterfall, one class-3 case (minus glyph fixed, no title) ----------
idx = int(np.where(y == 3)[0][0])
ev = explainer.expected_value
base_val = ev[TARGET] if np.ndim(ev) else ev
one = shap.Explanation(values=sv[idx, :, TARGET], base_values=base_val,
                       data=Xt.iloc[idx].values, feature_names=labels)
plt.figure(figsize=(8, 5))
shap.plots.waterfall(one, show=False)
plt.tight_layout()
plt.savefig(os.path.join(DEST, "Figure_S2.png"), dpi=200, bbox_inches="tight"); plt.close()

print("case id used for S2 waterfall:", int(df["id"].iloc[idx]), "| true class:", int(y[idx]))
from PIL import Image
for f in ["Figure_S1.png", "Figure_S2.png"]:
    im = Image.open(os.path.join(DEST, f)); print(f, im.size)
print("DONE")
