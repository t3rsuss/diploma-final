from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

R = Path(__file__).resolve().parent.parent / "results"

HEADER = ["has_sections", "n_sections", "max_sec_entropy", "n_imports"]
ENTROPY = ["file_entropy", "blk_ent_mean", "blk_ent_std", "blk_ent_min",
           "blk_ent_max", "frac_high_ent", "zero_frac"]
HIST = [f"h{i}" for i in range(256)]
SETS = {
    "заголовки": HEADER,
    "ентропія": ENTROPY,
    "ентропія+байти": ENTROPY + HIST,
    "усе": HEADER + ENTROPY + HIST,
}


def run(fmt):
    df = pd.concat([pd.read_csv(R / f"feat_{fmt}_clean.csv"),
                    pd.read_csv(R / f"feat_{fmt}_mal.csv")], ignore_index=True)
    meta = pd.read_csv(R / f"mal_{fmt}_meta.csv")
    sig = dict(zip(meta["sha256"], meta["signature"].fillna("")))

    def group(row):
        if row["label"] == "malware":
            s = sig.get(row["file"].replace(".bin", ""), "")
            return s if s else row["file"]
        return row["file"]

    groups = df.apply(group, axis=1)
    y = (df["label"] == "malware").astype(int).values
    print(f"\n=== {fmt.upper()} (групи за родиною): чистих {(y == 0).sum()}, шкідливих {y.sum()}")
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for name, cols in SETS.items():
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        p = cross_val_predict(clf, df[cols].values, y, cv=cv, groups=groups,
                              method="predict_proba")[:, 1]
        pred = (p >= 0.5).astype(int)
        print(f"{name:<22} AUC={roc_auc_score(y, p):.3f} "
              f"AP={average_precision_score(y, p):.3f} "
              f"P={precision_score(y, pred, zero_division=0):.3f} "
              f"R={recall_score(y, pred):.3f} F1={f1_score(y, pred):.3f}")


run("pe")
run("elf")