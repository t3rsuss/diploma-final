import pathlib
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
ARCHS = ["armel", "armhf", "arm64", "i386", "mipsel", "mips64el", "ppc64el"]


def flags(sc, st, keys):
    d = sc.merge(st[keys + ["agree"]], on=keys, how="left")
    d["r1"] = d["agree"].isna()
    d["r2"] = d["agree"] < 0.6
    d["rule"] = d["r1"] | d["r2"]
    d["std"] = d["level"] != "low"
    d["elev"] = d["score"] >= 0.5
    d["std_rule"] = d["std"] | d["rule"]
    d["elev_rule"] = d["elev"] | d["rule"]
    return d


parts = []
for a in ARCHS:
    d = flags(pd.read_csv(R / f"multi_{a}_scores.csv"),
              pd.read_csv(R / f"multi_{a}_struct.csv"), ["file", "sha256"])
    d["arch"] = a
    parts.append(d)
clean = pd.concat(parts, ignore_index=True)

print("=== Чисті ELF за архітектурами ===")
g = clean.groupby("arch").agg(файлів=("std", "size"), стандартний=("std", "sum"),
                              підвищений=("elev", "sum"), R1=("r1", "sum"),
                              R2=("r2", "sum"), правила=("rule", "sum"))
print(g.to_string())

mal = flags(pd.read_csv(R / "mal_elf_scores.csv"),
            pd.read_csv(R / "struct_mal_elf.csv"), ["sha256"])
ar = pd.read_csv(R / "mal_elf_arch.csv")
ar["sha256"] = ar["file"].str.replace(".bin", "", regex=False)
mal = mal.merge(ar[["sha256", "arch", "endian"]], on="sha256", how="left")


def mal_group(a, e):
    if a == "ARM":
        return "ARM"
    if a == "MIPS" and e == "LE":
        return "MIPS LE"
    if a == "x86":
        return "x86 32-біт"
    if a == "AArch64":
        return "AArch64"
    if a == "x86-64":
        return "x86-64"
    return "без чистого еталона"


mal["group"] = [mal_group(a, e) for a, e in zip(mal["arch"], mal["endian"])]
clean["group"] = clean["arch"].map({
    "armel": "ARM", "armhf": "ARM", "mipsel": "MIPS LE", "mips64el": "MIPS LE",
    "i386": "x86 32-біт", "arm64": "AArch64", "ppc64el": "PPC64 LE"})


def pct(s):
    return f"{int(s.sum())}/{len(s)} ({s.mean() * 100:.1f} %)" if len(s) else "-"


print("\n=== Шкідливі проти чистих за групами ===")
for grp in ["x86-64", "ARM", "MIPS LE", "x86 32-біт", "AArch64", "без чистого еталона"]:
    m = mal[mal["group"] == grp]
    c = clean[clean["group"] == grp]
    print(f"\n{grp}: шкідливих {len(m)}, чистих {len(c)}")
    for name, col in (("стандартний", "std"), ("стандартний + правила", "std_rule"),
                      ("підвищений", "elev"), ("підвищений + правила", "elev_rule")):
        print(f"  {name:24s} знайдено {pct(m[col]):16s} хибних {pct(c[col])}")

hit = clean[clean["std"] | clean["elev"] | clean["rule"]].sort_values("score", ascending=False)
print("\n=== Чисті файли, зачеплені детектором або правилами (до 40) ===")
print(hit[["arch", "file", "score", "agree", "std", "elev", "rule"]].head(40).to_string(index=False)
      if len(hit) else "жодного")