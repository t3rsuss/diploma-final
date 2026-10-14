import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"

mal = pd.concat([pd.read_csv(R / "mal_pe_scores.csv"),
                 pd.read_csv(R / "mal_elf_scores.csv")], ignore_index=True)
clean = pd.read_csv(R / "own_detector_clean.csv")


def tpr_no_fp(ms, cs):
    return float((ms > cs.max()).mean())


def tpr_at_fpr(ms, cs, fpr):
    thr = np.quantile(cs, 1 - fpr)
    return float((ms > thr).mean())


for key in ("win64", "elf"):
    m = mal[mal["model"] == key]
    c = clean[clean["model"] == key]
    print(f"\n=== {key}: шкідливих {len(m)}, чистих {len(c)} ===")
    for name, col in (("власна", "score"), ("готова EMBER2024", "score_pre")):
        ms, cs = m[col].to_numpy(), c[col].to_numpy()
        y = np.r_[np.zeros(len(cs)), np.ones(len(ms))]
        auc = roc_auc_score(y, np.r_[cs, ms])
        print(f"  {name:18s} AUC {auc:.4f} | знайдено без жодної хибної тривоги "
              f"{tpr_no_fp(ms, cs) * 100:.1f} % | при FPR 1 % "
              f"{tpr_at_fpr(ms, cs, 0.01) * 100:.1f} % | найвища оцінка чистих {cs.max():.3f}")

print("\n=== Пропущені власною моделлю x64-зразки: обидві оцінки ===")
w = mal[(mal["model"] == "win64") & (mal["level"] == "low")]
print(w[["file", "score", "score_pre"]].to_string(index=False))

print("\n=== ELF: пропущені власною моделлю, оцінка готової ===")
e = mal[(mal["model"] == "elf") & (mal["level"] == "low")].sort_values("score")
print(e[["file", "score", "score_pre"]].to_string(index=False))