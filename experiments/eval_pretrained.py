import sys
import pathlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import thrember
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.thrember_patch import apply
apply()

MODELS = ROOT / "data" / "ember_models"
LIMIT = 30_000_000

JOBS = [
    ("PE", "samples/clean", "EMBER2024_PE.model"),
    ("ELF", "samples/elf_clean", "EMBER2024_ELF.model"),
]

for name, folder, model_file in JOBS:
    model = lgb.Booster(model_file=str(MODELS / model_file))
    rows, skipped, big = [], 0, 0
    files = [p for p in sorted((ROOT / folder).rglob("*")) if p.is_file()]
    for p in tqdm(files, desc=name):
        if p.stat().st_size > LIMIT:
            big += 1
            continue
        try:
            score = thrember.predict_sample(model, p.read_bytes())
        except Exception as e:
            skipped += 1
            print("пропущено", p.name, e)
            continue
        rows.append({"file": p.name, "size": p.stat().st_size, "score": float(score)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / f"pretrained_{name.lower()}_clean.csv", index=False)
    s = df["score"].to_numpy()
    print(f"\n{name}: оброблено {len(s)}, помилок {skipped}, більших за 30 МБ {big}")
    print(f"середня оцінка {s.mean():.3f}, медіана {np.median(s):.3f}")
    for t in (0.3, 0.5, 0.8):
        print(f"  оцінка вище {t}: {(s > t).sum()} файлів ({(s > t).mean() * 100:.1f} %)")
    print(df.sort_values("score", ascending=False).head(8).to_string(index=False))