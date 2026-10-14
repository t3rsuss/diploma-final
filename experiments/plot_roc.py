import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
mal = pd.read_csv(R / "malware_scores_merged.csv")
clean = pd.read_csv(R / "own_detector_clean.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, fmt in zip(axes, ("PE", "ELF")):
    c = clean[clean["kind"] == fmt]
    m = mal[mal["kind"] == fmt]
    y = np.r_[np.zeros(len(c)), np.ones(len(m))]
    s = np.r_[c["score"].to_numpy(), m["score"].to_numpy()]
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    ax.plot(fpr * 100, tpr * 100, label=f"AUC = {auc:.4f}")
    fc = (c["level"] != "low").mean() * 100
    det = (m["level"] != "low").mean() * 100
    ax.scatter([fc], [det], color="red", zorder=5, label="робочий поріг системи")
    ax.set_xlabel("Частка хибних тривог на чистих, %")
    ax.set_ylabel("Частка знайдених шкідливих, %")
    ax.set_title(f"{fmt}: {len(m)} шкідливих, {len(c)} чистих")
    ax.set_xlim(-0.2, 5)
    ax.set_ylim(0, 101)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    print(fmt)
    for target in (0.0, 0.005, 0.01):
        i = np.searchsorted(fpr, target, side="right") - 1
        print(f"  при частці хибних тривог не більше {target * 100:.1f} %: "
              f"знайдено {tpr[max(i, 0)] * 100:.1f} %")
fig.tight_layout()
fig.savefig(R / "roc_own_malware.png", dpi=200)
print("збережено results\\roc_own_malware.png")