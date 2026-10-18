import pathlib
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
X86 = {"x86", "x86-64"}
CLEAN_MAP = {"armel": "ARM", "armhf": "ARM", "arm64": "AArch64", "i386": "x86",
             "mipsel": "MIPS", "mips64el": "MIPS", "ppc64el": "PowerPC"}
VARS = [("стандартний", "std"), ("стандартний + правила", "std_rule"),
        ("підвищений", "elev"), ("підвищений + правила", "elev_rule"),
        ("максимальний", "max"), ("максимальний + правила", "max_rule")]
COLS = ["sha256", "score", "level", "agree", "arch"]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(c - m, 0.0), min(c + m, 1.0)


def finish(d):
    d = d[COLS].copy()
    r1 = d["agree"].isna()
    r2 = d["arch"].isin(X86) & (d["agree"] < 0.6)
    d["rule"] = r1 | r2
    d["std"] = d["level"] != "low"
    d["elev"] = d["score"] >= 0.5
    d["max"] = d["score"] >= 0.2
    for c in ("std", "elev", "max"):
        d[c + "_rule"] = d[c] | d["rule"]
    return d


own = pd.read_csv(R / "own_detector_clean.csv")
own = own[own["kind"] == "ELF"].merge(
    pd.read_csv(R / "struct_clean_elf.csv")[["file", "sha256", "agree"]],
    on="file", how="left")
own["arch"] = "x86-64"

ex = pd.read_csv(R / "extra_scores.csv").merge(
    pd.read_csv(R / "struct_extra.csv")[["file", "sha256", "agree"]],
    on=["file", "sha256"], how="left")
ex["arch"] = "x86-64"

parts = [own[COLS], ex[COLS]]
for a, name in CLEAN_MAP.items():
    d = pd.read_csv(R / f"multi_{a}_scores.csv").merge(
        pd.read_csv(R / f"multi_{a}_struct.csv")[["file", "sha256", "agree"]],
        on=["file", "sha256"], how="left")
    d["arch"] = name
    parts.append(d[COLS])

raw = pd.concat(parts, ignore_index=True)
dup = raw["sha256"].notna() & raw["sha256"].duplicated()
print(f"чистих ELF до видалення дублів: {len(raw)}, дублів: {int(dup.sum())}, "
      f"після: {int((~dup).sum())}")
clean = finish(raw[~dup])

mal = pd.read_csv(R / "mal_elf_scores.csv").merge(
    pd.read_csv(R / "struct_mal_elf.csv")[["sha256", "agree"]], on="sha256", how="left")
ar = pd.read_csv(R / "mal_elf_arch.csv")
ar["sha256"] = ar["file"].str.replace(".bin", "", regex=False)
mal = finish(mal.merge(ar[["sha256", "arch"]], on="sha256", how="left"))


def cell(s):
    k, n = int(s.sum()), len(s)
    if n == 0:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n * 100:.1f} %, {lo * 100:.0f}-{hi * 100:.0f} %)"


def block(title, m, c):
    print(f"\n{title}: шкідливих {len(m)}, чистих {len(c)}")
    for name, col in VARS:
        print(f"  {name:24s} знайдено {cell(m[col]):30s} хибних {cell(c[col])}")


for arch in ["x86-64", "x86", "ARM", "MIPS", "AArch64", "PowerPC", "m68k", "SuperH"]:
    block(arch, mal[mal["arch"] == arch], clean[clean["arch"] == arch])
block("Усі, крім x86-64", mal[mal["arch"] != "x86-64"], clean[clean["arch"] != "x86-64"])
block("Усі ELF", mal, clean)

print("\nПропущені шкідливі (підвищений + правила):")
print(mal[~mal["elev_rule"]][["score", "level", "agree", "arch"]].to_string())
print("\nХибні тривоги (підвищений):")
print(clean[clean["elev"]][["score", "level", "agree", "arch"]].to_string())