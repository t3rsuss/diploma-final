import sys
import pathlib
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.structure.blocks import CLASSES, FEATURE_NAMES

d = np.load(ROOT / "results" / "blocks.npz")
X, y, g, fm = d["X"], d["y"], d["groups"], d["fmt"]

tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))
print("блоків навчання:", len(tr), "| тест:", len(te))

SETS = {
    "лише ентропія": [64],
    "один блок": list(range(72)),
    "блок + сусіди": list(range(216)),
    "блок + сусіди + позиція": list(range(217)),
}

best = None
for name, cols in SETS.items():
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63,
                             random_state=42, verbose=-1)
    clf.fit(X[tr][:, cols], y[tr])
    pred = clf.predict(X[te][:, cols])
    acc = accuracy_score(y[te], pred)
    f1 = f1_score(y[te], pred, average="macro")
    print(f"{name:26s} accuracy={acc:.3f}  macro-F1={f1:.3f}")
    best = (clf, pred, cols)

clf, pred, cols = best
labels = list(range(len(CLASSES)))
print(classification_report(y[te], pred, labels=labels,
                            target_names=CLASSES, zero_division=0))

for code, nm in ((0, "PE"), (1, "ELF")):
    m = fm[te] == code
    print(f"{nm}: accuracy={accuracy_score(y[te][m], pred[m]):.3f} "
          f"macro-F1={f1_score(y[te][m], pred[m], average='macro'):.3f}")

cm = confusion_matrix(y[te], pred, labels=labels)
cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
np.savetxt(ROOT / "results" / "blocks_confusion.csv", cm, fmt="%d", delimiter=",")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(labels); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_yticks(labels); ax.set_yticklabels(CLASSES)
for i in labels:
    for j in labels:
        ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
ax.set_xlabel("Передбачений тип"); ax.set_ylabel("Справжній тип")
fig.colorbar(im); fig.tight_layout()
fig.savefig(ROOT / "results" / "blocks_confusion.png", dpi=200)

(ROOT / "src" / "models").mkdir(parents=True, exist_ok=True)
joblib.dump({"model": clf, "classes": CLASSES,
             "features": [FEATURE_NAMES[i] for i in cols]},
            ROOT / "src" / "models" / "block_clf.joblib")
print("модель збережено")