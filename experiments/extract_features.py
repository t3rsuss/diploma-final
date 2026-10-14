import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.features.extract import extract

folder = ROOT / sys.argv[1]
label = sys.argv[2]
out = ROOT / "results" / sys.argv[3]

files = sorted(p for p in folder.iterdir() if p.is_file())
rows = []
for i, f in enumerate(files, 1):
    try:
        row = extract(f)
    except Exception as exc:
        print("помилка", f.name, exc)
        continue
    row["label"] = label
    rows.append(row)
    if i % 100 == 0:
        print(i, "з", len(files))

df = pd.DataFrame(rows)
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print("Збережено:", len(df), "рядків,", df.shape[1], "колонок")