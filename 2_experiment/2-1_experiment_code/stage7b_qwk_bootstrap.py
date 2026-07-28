# -*- coding: utf-8 -*-
"""Stage 7b - Ordinal-aware paired comparison (RQ2).
Case-bootstrap 95% CI for the QWK difference between the best ML model (SVM + TF-IDF, OOF
predictions) and each LLM configuration on the shared 1,500 cases. Backs the manuscript claim
that ML's QWK advantage holds against the zero-shot models and Qwen few-shot, while GPT few-shot
is statistically on par.

Inputs : outputs/stage3_oof_predictions.parquet, outputs/stage6_llm_predictions.parquet
Outputs: outputs/stage7b_qwk_bootstrap.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import config
config.set_seed()
OUT = config.OUTPUT_DIR
LABELS = [0, 1, 2, 3]
B = 10000

def qwk(yt, yp):
    return cohen_kappa_score(yt, yp, labels=LABELS, weights="quadratic")

oof = pd.read_parquet(OUT / "stage3_oof_predictions.parquet")
ml = (oof[(oof.model == "svm") & (oof.rep == "tfidf")][["id", "y_true", "y_pred"]]
      .drop_duplicates("id").set_index("id"))
llm = pd.read_parquet(OUT / "stage6_llm_predictions.parquet")
CONFIGS = [("gpt", "zero_shot_closed_book", "GPT zero-shot"),
           ("qwen", "zero_shot_closed_book", "Qwen zero-shot"),
           ("gpt", "few_shot_open_book", "GPT few-shot"),
           ("qwen", "few_shot_open_book", "Qwen few-shot")]
rng = np.random.RandomState(config.SEED)
rows = []
for m, v, disp in CONFIGS:
    g = (llm[(llm.model == m) & (llm.variant == v)][["id", "y_pred"]]
         .drop_duplicates("id").set_index("id"))
    ids = ml.index.intersection(g.index)
    yt = ml.loc[ids, "y_true"].astype(int).values
    mlp = ml.loc[ids, "y_pred"].astype(int).values
    lp = g.loc[ids, "y_pred"].astype(int).values
    q_ml, q_llm = qwk(yt, mlp), qwk(yt, lp)
    n = len(ids)
    diffs = np.empty(B)
    for b in range(B):
        s = rng.randint(0, n, n)
        diffs[b] = qwk(yt[s], mlp[s]) - qwk(yt[s], lp[s])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    excl = bool(lo > 0 or hi < 0)
    rows.append(dict(config=disp, n=n, qwk_ml=round(q_ml, 3), qwk_llm=round(q_llm, 3),
                     dqwk=round(q_ml - q_llm, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                     excludes_zero=excl))
    print(f"{disp:16s} n={n}  QWK ML {q_ml:.3f} vs LLM {q_llm:.3f}  "
          f"dQWK {q_ml-q_llm:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"{'excludes 0' if excl else 'INCLUDES 0 (on par)'}")
pd.DataFrame(rows).to_csv(OUT / "stage7b_qwk_bootstrap.csv", index=False, encoding="utf-8-sig")
print("\nStage 7b done -> outputs/stage7b_qwk_bootstrap.csv")
