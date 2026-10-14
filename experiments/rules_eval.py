import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"

mal = pd.concat([pd.read_csv(R / "mal_pe_scores.csv"),
                 pd.read_csv(R / "mal_elf_scores.csv")], ignore_index=True)
clean = pd.read_csv(R / "own_detector_clean.csv")
st_mal = pd.concat([pd.read_csv(R / "struct_mal_pe.csv"),
                    pd.read_csv(R / "struct_mal_elf.csv")], ignore_index=True)
st_clean = pd.concat([pd.read_csv(R / "struct_clean_pe.csv"),
                      pd.read_csv(R / "struct_clean_elf.csv")], ignore_index=True)

cl = clean.merge(st_clean.drop(columns=["sha256"]), on=["file", "kind"], how="left")
cl["label"] = 0
cl["has_table"] = cl["agree"].notna()

m = mal.merge(st_mal.drop(columns=["file", "kind"]), on="sha256", how="left", indicator=True)
m["label"] = 1
m["has_table"] = m["_merge"] == "both"

df = pd.concat([cl, m.drop(columns=["_merge"])], ignore_index=True)

own = df["model"].isin(["win64", "elf"])
std = (df["level"] != "low").to_numpy()
elev = np.where(own, df["score"] >= 0.5, df["level"] != "low")

is_elf = df["kind"] == "ELF"
r1 = is_elf & ~df["has_table"]
r2 = is_elf & df["has_table"] & (df["agree"] < 0.6)
r3 = df["has_table"] & (df["code_sections_not_code"] > 0.2)
r4 = df["has_table"] & (df["share_ent_gt7"] > 0.5)

rules = {"R1 ELF без таблиці секцій": r1,
         "R2 ELF, збіг < 0,6": r2,
         "R3 (лише довідково) секції коду не схожі на код > 20 %": r3,
         "R4 (лише довідково) ентропія > 7 у понад 50 % блоків": r4}
any_rule = (r1 | r2).to_numpy()

lab = df["label"].to_numpy()
fmt = df["kind"].to_numpy()

print("=== Окремі правила: скільки файлів зачіпає ===")
for f in ("PE", "ELF"):
    print(f"\n{f}:")
    for name, r in rules.items():
        r = r.to_numpy()
        c = int((r & (lab == 0) & (fmt == f)).sum())
        mm = int((r & (lab == 1) & (fmt == f)).sum())
        print(f"  {name:56s} чистих {c} з {int(((lab == 0) & (fmt == f)).sum())}, "
              f"шкідливих {mm} з {int(((lab == 1) & (fmt == f)).sum())}")

print("\n=== R1 або R2 (ELF): підсумок ===")
c = int((any_rule & (lab == 0)).sum())
mm = int((any_rule & (lab == 1)).sum())
print(f"  чистих ELF {c} з {int(((lab == 0) & (fmt == 'ELF')).sum())}, "
      f"шкідливих ELF {mm} з {int(((lab == 1) & (fmt == 'ELF')).sum())}")

print("\n=== Детектор і правила разом ===")
variants = [("детектор, стандартний режим", std),
            ("детектор + R1/R2, стандартний", std | any_rule),
            ("детектор, підвищений режим", elev),
            ("детектор + R1/R2, підвищений", elev | any_rule)]
for f in ("PE", "ELF"):
    nm = int(((lab == 1) & (fmt == f)).sum())
    nc = int(((lab == 0) & (fmt == f)).sum())
    print(f"\n{f}: шкідливих {nm}, чистих {nc}")
    for name, flag in variants:
        flag = np.asarray(flag)
        tp = int((flag & (lab == 1) & (fmt == f)).sum())
        fp = int((flag & (lab == 0) & (fmt == f)).sum())
        print(f"  {name:34s} знайдено {tp} ({tp / nm * 100:.1f} %), "
              f"хибних тривог {fp} ({fp / nc * 100:.2f} %)")

print("\n=== Шкідливі ELF, які знайшли тільки правила (стандартний режим) ===")
only = df[(df["label"] == 1) & any_rule & ~std]
cols = ["file", "kind", "model", "signature", "score", "agree", "has_table"]
cols = [c for c in cols if c in only.columns]
print(only[cols].to_string(index=False) if len(only) else "жодного")

print("\n=== Чисті файли, зачеплені R1/R2 ===")
hit = df[(df["label"] == 0) & any_rule]
cols = ["file", "kind", "model", "score", "agree", "has_table"]
print(hit[cols].to_string(index=False) if len(hit) else "жодного")