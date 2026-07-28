# -*- coding: utf-8 -*-
"""
Figure 2 - Error geometry and LLM bias.
Row-normalised (recall) confusion matrices for the best ML model (SVM + TF-IDF, out-of-fold) and the
two zero-shot LLMs, making the opposite collapse patterns visible: GPT-5 onto the minority short
class (1), Qwen3-8B onto the majority medium class (2). A predicted-class share strip sits under each
panel against the true distribution.

Inputs : outputs/stage3_oof_predictions.parquet, outputs/stage6_llm_predictions.parquet
Output : 4_figures/Figure_2.png (overwrites the released copy)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

import config
OUT = config.OUTPUT_DIR
LABELS = [0, 1, 2, 3]
CLASS_NAMES = ["0\nfine/none", "1\nshort", "2\nmedium", "3\nlong"]

oof = pd.read_parquet(OUT / "stage3_oof_predictions.parquet")
llm = pd.read_parquet(OUT / "stage6_llm_predictions.parquet")

ml = oof[(oof.model == "svm") & (oof.rep == "tfidf")][["y_true", "y_pred"]]
gpt = llm[(llm.model == "gpt") & (llm.variant == "zero_shot_closed_book")][["y_true", "y_pred"]]
qwen = llm[(llm.model == "qwen") & (llm.variant == "zero_shot_closed_book")][["y_true", "y_pred"]]

panels = [("SVM + TF-IDF (best ML)", ml),
          ("GPT-5, zero-shot", gpt),
          ("Qwen3-8B, zero-shot", qwen)]

y_true_dist = np.bincount(ml.y_true, minlength=4) / len(ml)  # same 1,500 truth for all

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
fig = plt.figure(figsize=(11, 4.3))
gs = GridSpec(2, 3, height_ratios=[5, 1.15], hspace=0.45, wspace=0.3,
              left=0.07, right=0.97, top=0.86, bottom=0.10)

for j, (name, d) in enumerate(panels):
    yt, yp = d.y_true.to_numpy(), d.y_pred.to_numpy()
    cm = confusion_matrix(yt, yp, labels=LABELS).astype(float)
    rn = cm / cm.sum(axis=1, keepdims=True)  # row-normalised = recall
    mf = f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)
    qwk = cohen_kappa_score(yt, yp, labels=LABELS, weights="quadratic")

    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_title(f"{name}\nmacro-F1 {mf:.3f} · QWK {qwk:.3f}")
    ax.set_xticks(range(4)); ax.set_xticklabels([c.split('\n')[0] for c in CLASS_NAMES])
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES if j == 0 else [c.split('\n')[0] for c in CLASS_NAMES])
    ax.set_xlabel("predicted")
    if j == 0:
        ax.set_ylabel("true")
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f"{rn[a, b]:.2f}", ha="center", va="center",
                    color="white" if rn[a, b] > 0.5 else "black", fontsize=9)

    # predicted-share strip vs true distribution
    axb = fig.add_subplot(gs[1, j])
    pred_dist = np.bincount(yp[(yp >= 0)], minlength=4) / len(yp)
    x = np.arange(4)
    axb.bar(x - 0.2, y_true_dist, width=0.38, color="0.7", label="true")
    axb.bar(x + 0.2, pred_dist, width=0.38, color="#2b6cb0", label="predicted")
    axb.set_xticks(x); axb.set_xticklabels([c.split('\n')[0] for c in CLASS_NAMES])
    axb.set_ylim(0, 1); axb.set_yticks([0, 0.5, 1.0])
    axb.set_ylabel("share" if j == 0 else "")
    if j == 0:
        axb.legend(fontsize=8, loc="upper left", frameon=False)

cbar = fig.colorbar(im, ax=fig.axes[0::2], fraction=0.015, pad=0.02)
cbar.set_label("row-normalised rate (recall)")
fig.suptitle("Error geometry: ML predicts across classes; zero-shot LLMs collapse in opposite directions",
             y=0.975, fontsize=11)

out = config.BASE_DIR / "4_figures" / "Figure_2.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
for name, d in panels:
    pd_dist = np.bincount(d.y_pred.to_numpy().clip(0), minlength=4) / len(d)
    print(f"  {name:28s} predicted share = {np.round(pd_dist,2)}")
