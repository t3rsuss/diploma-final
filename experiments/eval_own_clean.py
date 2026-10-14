import sys
import pathlib
import pandas as pd
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detector.verdict import score_file, level

JOBS = [("PE", "samples/clean"), ("ELF", "samples/elf_clean")]
LIMIT = 30_000_000
rows = []
for name, folder in JOBS:
    files = [p for p in sorted((ROOT / folder).rglob("*"))
             if p.is_file() and p.stat().st_size <= LIMIT]
    for p in tqdm(files, desc=name):
        try:
            r = score_file(str(p))
        except Exception as e:
            print("пропущено", p.name, e)
            continue
        if r is None:
            continue
        rows.append(dict(file=p.name, kind=r["kind"], model=r["key"],
                         score=r["score"], score_pre=r["score_pre"],
                         prob=r.get("prob"), level=level(r)))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "own_detector_clean.csv", index=False)
print(df.groupby(["model", "level"]).size().unstack(fill_value=0))
for key, g in df.groupby("model"):
    flagged = (g["level"] != "low")
    print(f"{key}: файлів {len(g)}, підозрілий або шкідливий {flagged.sum()} "
          f"({flagged.mean() * 100:.2f} %), шкідливий {(g['level'] == 'high').sum()}")
print(df.sort_values("score", ascending=False).head(10).to_string(index=False))