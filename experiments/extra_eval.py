import pathlib
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
sc = pd.read_csv(R / "extra_scores.csv")
st = pd.read_csv(R / "struct_extra.csv")
pk = pd.read_csv(R / "packed_scores.csv")

print("колонки scores:", list(sc.columns))
print("колонки struct:", list(st.columns))
print("колонки packed:", list(pk.columns))

key = "file" if ("file" in sc.columns and "file" in st.columns) else "sha256"
for d in (sc, st):
    if key == "file":
        d[key] = d[key].astype(str).map(lambda p: pathlib.Path(p).name)

drop = [c for c in ("kind", "sha256") if c in st.columns and c in sc.columns and c != key]
df = st.drop(columns=drop).merge(sc, on=key, how="outer", indicator=True)

print(f"\nфайлів у struct: {len(st)}, оцінено детектором: {len(sc)}")
noscore = df[df["_merge"] == "left_only"]
print("не оцінено детектором:", list(noscore[key]) if len(noscore) else "немає")

df = df[df["_merge"] != "left_only"].copy()
model = df["model"] if "model" in df.columns else pd.Series("elf", index=df.index)
score = df["score"] if "score" in df.columns else pd.Series(np.nan, index=df.index)
level = df["level"] if "level" in df.columns else pd.Series("low", index=df.index)

std = (level != "low").to_numpy()
elev = np.where(model.isin(["win64", "elf"]), score >= 0.5, level != "low")
has = df["agree"].notna() if "agree" in df.columns else pd.Series(False, index=df.index)
r1 = ~has
r2 = has & (df["agree"] < 0.6)
r3 = has & (df["code_sections_not_code"] > 0.2)
elf_rules = (r1 | r2).to_numpy()

n = len(df)
print(f"\n=== Незалежні чисті файли: {n} ===")
print(f"детектор, стандартний: хибних {int(std.sum())} ({std.mean() * 100:.1f} %)")
print(f"детектор, підвищений:  хибних {int(elev.sum())} ({elev.mean() * 100:.1f} %)")
print(f"R1 (без таблиці): {int(r1.sum())}, R2 (збіг < 0,6): {int(r2.sum())}, "
      f"R3 (інформаційне): {int(r3.sum())}")
print(f"R1 або R2: {int(elf_rules.sum())} ({elf_rules.mean() * 100:.1f} %)")

cols = [c for c in (key, "model", "score", "level", "agree", "code_sections_not_code") if c in df.columns]
flag = std | np.asarray(elev) | elf_rules | r3.to_numpy()
print("\n=== Файли, зачеплені детектором або правилами ===")
print(df.loc[flag, cols].to_string(index=False) if flag.any() else "жодного")

print("\n=== Упаковані UPX контрольні файли ===")
print(pk.to_string(index=False))
print("У struct_packed.csv 0 файлів: таблицю секцій не прочитано, тож R1 спрацьовує.")