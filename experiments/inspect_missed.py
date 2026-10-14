import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"

mal = pd.concat([pd.read_csv(R / "mal_pe_scores.csv"),
                 pd.read_csv(R / "mal_elf_scores.csv")], ignore_index=True)
st = pd.concat([pd.read_csv(R / "struct_mal_pe.csv"),
                pd.read_csv(R / "struct_mal_elf.csv")], ignore_index=True)
cl = pd.concat([pd.read_csv(R / "struct_clean_pe.csv"),
                pd.read_csv(R / "struct_clean_elf.csv")], ignore_index=True)
meta = pd.concat([pd.read_csv(R / "mal_pe_meta.csv"),
                  pd.read_csv(R / "mal_elf_meta.csv")], ignore_index=True)[["sha256", "signature"]]

m = mal.merge(st.drop(columns=["file", "kind"]), on="sha256", how="left", indicator=True)
m = m.merge(meta, on="sha256", how="left")
m["signature"] = m["signature"].fillna("(не визначено)")

print("=== Зразки без читаної таблиці секцій ===")
nt = m[m["_merge"] == "left_only"]
print(f"усього {len(nt)} з {len(m)}")
print(nt.groupby(["kind", "level"]).size().to_string())
print(nt[["file", "kind", "model", "signature", "score", "level"]]
      .assign(file=lambda d: d["file"].str[:12])
      .sort_values("score").to_string(index=False))

print("\n=== Пропущені детектором зразки з прочитаною структурою ===")
print("(у дужках частка чистих файлів того ж формату, які не менш підозрілі за цією ознакою)")
feats = [("agree", "менше"), ("code_sections_not_code", "більше"),
         ("share_ent_gt7", "більше"), ("packed_code_share", "більше")]
missed = m[(m["level"] == "low") & (m["_merge"] == "both")]
for _, r in missed.iterrows():
    c = cl[cl["kind"] == r["kind"]]
    parts = []
    for f, direction in feats:
        v = r[f]
        frac = (c[f] <= v).mean() if direction == "менше" else (c[f] >= v).mean()
        parts.append(f"{f}={v:.3f} ({frac * 100:.1f} %)")
    print(f"{r['file'][:12]} {r['kind']} {r['signature']}, оцінка {r['score']:.3f}: "
          + "; ".join(parts))