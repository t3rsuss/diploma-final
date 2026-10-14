import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.parsing.parser import parse_binary

folder = ROOT / sys.argv[1]
label = sys.argv[2]
out = ROOT / "results" / sys.argv[3]

files = sorted(p for p in folder.iterdir() if p.is_file())
rows = []
for i, f in enumerate(files, 1):
    info = parse_binary(f)
    secs = info["sections"]
    rows.append({
        "file": info["file"],
        "label": label,
        "format": info["format"],
        "size": info["size"],
        "file_entropy": info["file_entropy"],
        "n_sections": len(secs),
        "max_sec_entropy": max((s["entropy"] for s in secs), default=0),
        "n_imports": len(info["imports"]),
        "error": info["error"],
    })
    if i % 100 == 0:
        print(i, "з", len(files))

df = pd.DataFrame(rows)
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(df["format"].value_counts(dropna=False))
print("Помилок:", df["error"].notna().sum())
print(df[["size", "file_entropy", "n_sections", "n_imports"]].describe().round(2))