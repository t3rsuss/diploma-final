import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"

clean_sc = pd.read_csv(R / "own_detector_clean.csv")
mal_sc = pd.concat([pd.read_csv(R / "mal_pe_scores.csv"),
                    pd.read_csv(R / "mal_elf_scores.csv")], ignore_index=True)
st_clean = pd.concat([pd.read_csv(R / "struct_clean_pe.csv"),
                      pd.read_csv(R / "struct_clean_elf.csv")], ignore_index=True)
st_mal = pd.concat([pd.read_csv(R / "struct_mal_pe.csv"),
                    pd.read_csv(R / "struct_mal_elf.csv")], ignore_index=True)
meta = pd.concat([pd.read_csv(R / "mal_pe_meta.csv"),
                  pd.read_csv(R / "mal_elf_meta.csv")], ignore_index=True)[["sha256", "signature"]]

STRUCT = [c for c in st_clean.columns if c not in ("file", "sha256", "kind", "n_blocks")]

c = clean_sc[["file", "kind", "score"]].merge(
    st_clean.drop(columns=["sha256"]), on=["file", "kind"])
c["label"] = 0
c["group"] = ["clean_" + str(i) for i in range(len(c))]

m = mal_sc[["sha256", "kind", "score"]].merge(
    st_mal.drop(columns=["file", "kind"]), on="sha256").merge(meta, on="sha256", how="left")
m["label"] = 1
m["group"] = m["signature"].fillna(m["sha256"]).where(m["signature"].notna(), m["sha256"])

df = pd.concat([c, m], ignore_index=True)
df["score_logit"] = np.log(np.clip(df["score"], 1e-6, 1 - 1e-6) /
                           (1 - np.clip(df["score"], 1e-6, 1 - 1e-6)))


def cv_eval(d, cols, reps=20):
    X, y, g = d[cols].to_numpy(), d["label"].to_numpy(), d["group"].to_numpy()
    out = []
    for r in range(reps):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y, g):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        fpr, tpr, _ = roc_curve(y, oof)
        out.append((roc_auc_score(y, oof),
                    float(tpr[fpr <= 0.01].max()), float(tpr[fpr <= 0.05].max())))
    a = np.array(out)
    return a.mean(0), a.std(0)


for fmt in ("PE", "ELF"):
    d = df[df["kind"] == fmt].reset_index(drop=True)
    print(f"\n=== {fmt}: чистих {int((d.label == 0).sum())}, шкідливих {int((d.label == 1).sum())} ===")

    print("середні значення ознак (чисті | шкідливі):")
    show = ["agree", "code_outside_code_sections", "code_sections_not_code",
            "share_ent_gt7", "packed_code_share", "wx_sections", "overlay_ratio"]
    means = d.groupby("label")[show].mean().T
    means.columns = ["чисті", "шкідливі"]
    print(means.round(3).to_string())

    for name, cols in (("А: лише оцінка детектора", ["score_logit"]),
                       ("Б: лише структурні ознаки", STRUCT),
                       ("В: оцінка + структура", ["score_logit"] + STRUCT)):
        mu, sd = cv_eval(d, cols)
        print(f"{name:28s} AUC {mu[0]:.4f}±{sd[0]:.4f} | "
              f"TPR@FPR1% {mu[1] * 100:.1f}±{sd[1] * 100:.1f} | "
              f"TPR@FPR5% {mu[2] * 100:.1f}±{sd[2] * 100:.1f}")