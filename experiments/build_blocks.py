import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.structure.blocks import file_blocks, CLASSES

folders = sys.argv[1:] or ["samples/clean", "samples/elf_clean"]
X, y, g, fm = [], [], [], []
fid, seen = 0, 0

for folder in folders:
    for p in sorted((ROOT / folder).rglob("*")):
        if not p.is_file() or p.stat().st_size > 30_000_000:
            continue
        seen += 1
        try:
            res = file_blocks(p)
        except Exception as e:
            print("skip", p.name, e)
            continue
        if res is None:
            continue
        fmt, feats, labels = res
        X.append(feats)
        y.append(labels)
        g.append(np.full(len(labels), fid))
        fm.append(np.full(len(labels), 0 if fmt == "PE" else 1))
        fid += 1
        if fid % 100 == 0:
            print(f"оброблено файлів: {fid}")

X, y, g, fm = map(np.concatenate, (X, y, g, fm))
out = ROOT / "results" / "blocks.npz"
np.savez_compressed(out, X=X, y=y, groups=g, fmt=fm)
print("переглянуто файлів:", seen, "| взято:", fid, "| блоків:", len(y))
for i, name in enumerate(CLASSES):
    print(f"{name:10s} {(y == i).sum()}")