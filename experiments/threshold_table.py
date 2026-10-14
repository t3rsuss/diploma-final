import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"

mal = pd.concat([pd.read_csv(R / "mal_pe_scores.csv"),
                 pd.read_csv(R / "mal_elf_scores.csv")], ignore_index=True)
clean = pd.read_csv(R / "own_detector_clean.csv")

THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.9673]
rows = []
for key in ("win64", "elf"):
    m = mal[mal["model"] == key]["score"].to_numpy()
    c = clean[clean["model"] == key]["score"].to_numpy()
    for t in THRESHOLDS:
        rows.append(dict(
            модель=key, поріг=t,
            знайдено=f"{int((m >= t).sum())} з {len(m)}",
            TPR_відсотків=round((m >= t).mean() * 100, 1),
            хибних=f"{int((c >= t).sum())} з {len(c)}",
            FPR_відсотків=round((c >= t).mean() * 100, 2)))
df = pd.DataFrame(rows)
df.to_csv(R / "threshold_table.csv", index=False)
print(df.to_string(index=False))