# -*- coding: utf-8 -*-
"""Stage 8b - Robustness of the explanation-agreement result (RQ3).
(1) Case-bootstrap 95% CI for Kendall tau between r_ML (Stage 5 SHAP) and the LLM
    order-of-mention ranking, for the two independent zero-shot configs. Tests whether the
    tau=0.80 (GPT) / 1.00 (Qwen) point estimates are stable.
(2) Model-agnostic KernelSHAP ranking on the structured SVM (the SVM over the five sentencing
    factors), to check the factor ranking is not an artefact of the LightGBM/TreeSHAP choice.

Inputs : outputs/stage5_factor_ranking.csv, outputs/stage6_llm_predictions.parquet,
         outputs/stage2_features_verified.parquet
Outputs: outputs/stage8b_robustness.csv
"""
import sys, io, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import config
config.set_seed()
OUT = config.OUTPUT_DIR
B = 10000

FACTORS = ["prior_dui_count", "bac", "distance_km", "offense_year", "vehicle"]

# Korean keyword map: factor -> substrings that signal a mention in the LLM rationale
# (identical to the map in stage8_agreement.py). The LLMs answer in Korean, so these substrings
# must stay in Korean; they are the operational definition of "the model cited this factor".
# English meaning of every keyword, in the order the keywords are listed for each factor below:
#   prior_dui_count  "prior conviction / criminal record", "of the same kind", "repeat
#                    offense", "recidivism (statutory aggravation)", "first-time offender"
#   bac              "blood alcohol", "alcohol concentration", a variant spelling of "blood
#                    alcohol", then "the measured drinking reading" in its two spacings
#   distance_km      "driving distance" in its two spacings, then "distance"
#   offense_year     "year of the offense" in its two spacings, "year", "time of the offense"
#                    in its two spacings, "point in time of the offense"
#   vehicle          "vehicle type", "vehicle", "passenger (car)", "two-wheeled", "motorbike",
#                    "truck", "multi-passenger van", "bus", "taxi", Carnival (a widely sold
#                    Korean MPV model), "freight, cargo", "scooter", "moped"
KEYWORDS = {
    "prior_dui_count": ["전과", "동종", "재범", "누범", "초범"],
    "bac":             ["혈중알코올", "알코올농도", "혈중알콜", "음주수치", "음주 수치"],
    "distance_km":     ["운전거리", "운전 거리", "거리"],
    "offense_year":    ["범행연도", "범행 연도", "연도", "범행시기", "범행 시기", "범행시점"],
    "vehicle":         ["차종", "차량", "승용", "이륜", "오토바이", "트럭", "승합",
                         "버스", "택시", "카니발", "화물", "스쿠터", "원동기"],
}
YEAR_RE = re.compile(r"(19|20)\d{2}")

def _seg_factor(seg):
    for f, kws in KEYWORDS.items():
        if any(k in seg for k in kws):
            return f
    if YEAR_RE.search(seg):
        return "offense_year"
    return None

def order_positions(text):
    pos = {}
    for i, seg in enumerate(re.split(r"[,、/]", str(text))):
        f = _seg_factor(seg)
        if f is not None and f not in pos:
            pos[f] = i + 1
    return pos

# r_ML ranks (in FACTORS order)
ml = pd.read_csv(OUT / "stage5_factor_ranking.csv").set_index("factor")
r_ml_vec = np.array([int(ml.loc[f, "rank"]) for f in FACTORS])

def ranks_from_meanpos(meanpos):
    order = sorted(range(len(FACTORS)), key=lambda j: (meanpos[j], j))
    ranks = np.empty(len(FACTORS), dtype=int)
    for rank, j in enumerate(order, 1):
        ranks[j] = rank
    return ranks

llm = pd.read_parquet(OUT / "stage6_llm_predictions.parquet")
rows = []
print("=== (1) Order-of-mention Kendall tau, case-bootstrap ===")
for m, v, disp in [("gpt", "zero_shot_closed_book", "GPT zero-shot"),
                   ("qwen", "zero_shot_closed_book", "Qwen zero-shot")]:
    texts = llm[(llm.model == m) & (llm.variant == v)].factors.tolist()
    n = len(texts)
    # per-case position matrix (NaN if factor unmentioned) -- parse once
    pos_mat = np.full((n, len(FACTORS)), np.nan)
    for i, t in enumerate(texts):
        for f, p in order_positions(t).items():
            pos_mat[i, FACTORS.index(f)] = p
    def tau_for(idx):
        sub = pos_mat[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mp = np.nanmean(sub, axis=0)
        mp = np.where(np.isnan(mp), np.inf, mp)
        return kendalltau(ranks_from_meanpos(mp), r_ml_vec)[0]
    tau_point = tau_for(np.arange(n))
    rng = np.random.RandomState(config.SEED)
    taus = np.array([tau_for(rng.randint(0, n, n)) for _ in range(B)])
    taus = taus[~np.isnan(taus)]
    lo, hi = np.percentile(taus, [2.5, 97.5])
    rows.append(dict(analysis="order_tau", config=disp, point=round(tau_point, 2),
                     ci_lo=round(lo, 2), ci_hi=round(hi, 2)))
    print(f"  {disp:16s} tau={tau_point:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

# === (2) KernelSHAP on the structured SVM ===
print("\n=== (2) KernelSHAP ranking on the structured SVM (5 factors) ===")
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import shap

v = pd.read_parquet(OUT / "stage2_features_verified.parquet")
NUM = ["bac", "distance_km", "prior_dui_count", "offense_year"]
CAT = ["vehicle"]
X = v[NUM + CAT].copy()
for c in NUM:
    X[c] = pd.to_numeric(X[c], errors="coerce")
y = v["class4"].astype(int).values
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
])
Xt = pre.fit_transform(X)
names = NUM + list(pre.named_transformers_["cat"].get_feature_names_out(CAT))
svc = SVC(class_weight="balanced", probability=True, random_state=config.SEED).fit(Xt, y)
rng = np.random.RandomState(config.SEED)
bg = shap.sample(Xt, 50, random_state=config.SEED)
ev = shap.KernelExplainer(svc.predict_proba, bg)
samp = rng.choice(len(Xt), 120, replace=False)
sv = ev.shap_values(Xt[samp], nsamples=100, silent=True)
# sv: list (per class) of (n_samples, n_features); mean|.| over samples then classes
per_feat = np.stack([np.abs(s).mean(axis=0) for s in sv], axis=0).mean(axis=0)
imp = {}
for nme, val in zip(names, per_feat):
    base = "vehicle" if nme.startswith("vehicle_") else nme
    imp[base] = imp.get(base, 0.0) + val
ranked = sorted(imp, key=lambda k: -imp[k])
for i, f in enumerate(ranked, 1):
    print(f"  {i}. {f:18s} {imp[f]:.4f}")
    rows.append(dict(analysis="kernelshap_svm", config=f, point=round(imp[f], 4),
                     ci_lo="", ci_hi="", rank=i))
print(f"  -> KernelSHAP top-2: {ranked[:2]}")

pd.DataFrame(rows).to_csv(OUT / "stage8b_robustness.csv", index=False, encoding="utf-8-sig")
print("\nStage 8b done -> outputs/stage8b_robustness.csv")
