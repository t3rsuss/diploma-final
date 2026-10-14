import pathlib
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
sc = pd.read_csv(R / "mal_elf_scores.csv")
st = pd.read_csv(R / "struct_mal_elf.csv")[["sha256", "agree"]]
ar = pd.read_csv(R / "mal_elf_arch.csv")
ar["sha256"] = ar["file"].str.replace(".bin", "", regex=False)
meta = pd.read_csv(R / "mal_elf_meta.csv")[["sha256", "signature"]]

df = sc.merge(ar[["sha256", "arch", "bits", "endian"]], on="sha256", how="left")
df = df.merge(st, on="sha256", how="left").merge(meta, on="sha256", how="left")
df["signature"] = df["signature"].fillna("(не визначено)")
df["group"] = np.where(df["arch"] == "x86-64", "x86-64", "інші архітектури")

has_table = df["agree"].notna()
rule = ((~has_table) | (df["agree"] < 0.6)).to_numpy()
std = (df["level"] != "low").to_numpy()
elev = (df["score"] >= 0.5).to_numpy()

variants = [("лише детектор, стандартний", std),
            ("з правилами R1/R2, стандартний", std | rule),
            ("лише детектор, підвищений", elev),
            ("з правилами R1/R2, підвищений", elev | rule)]

print("=== Родини за групами архітектур ===")
print(pd.crosstab(df["signature"], df["group"]).to_string())

print("\n=== Знайдено шкідливих за групами ===")
for g in ("x86-64", "інші архітектури"):
    m = (df["group"] == g).to_numpy()
    n = int(m.sum())
    print(f"\n{g}: шкідливих {n}")
    for name, f in variants:
        k = int((f & m).sum())
        print(f"  {name:32s} {k:3d} ({k / n * 100:.1f} %)")

print("\n=== За архітектурами (стандартний: лише детектор / з правилами) ===")
for a, d in df.groupby("arch"):
    idx = d.index.to_numpy()
    print(f"{a:10s} {len(d):3d}   {int(std[idx].sum()):3d} / {int((std | rule)[idx].sum()):3d}")

print("\n=== Пропущено навіть у підвищеному режимі з правилами ===")
miss = df[~(elev | rule)]
cols = ["file", "arch", "signature", "score", "agree"]
print(miss[cols].assign(file=lambda d: d["file"].str[:12]).to_string(index=False)
      if len(miss) else "жодного")