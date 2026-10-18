import pathlib
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
ARCHS = ["armel", "armhf", "arm64", "i386", "mipsel", "mips64el", "ppc64el"]

sets = {"clean_elf_x64": "struct_clean_elf.csv", "extra_x64": "struct_extra.csv",
        "clean_pe": "struct_clean_pe.csv", "mal_elf": "struct_mal_elf.csv",
        "mal_pe": "struct_mal_pe.csv"}
sets.update({f"multi_{a}": f"multi_{a}_struct.csv" for a in ARCHS})

hashes = {}
for name, f in sets.items():
    d = pd.read_csv(R / f)
    hashes[name] = set(d["sha256"])
    print(f"{name:16s} рядків {len(d):5d}  унікальних sha256 {d['sha256'].nunique():5d}")

print("\nПерекриття:")
print("clean_elf_x64 і extra_x64:", len(hashes["clean_elf_x64"] & hashes["extra_x64"]))
print("clean_elf_x64 і mal_elf:", len(hashes["clean_elf_x64"] & hashes["mal_elf"]))
multi = set().union(*(hashes[f"multi_{a}"] for a in ARCHS))
print("multi і clean_elf_x64:", len(multi & hashes["clean_elf_x64"]))
print("унікальних у multi загалом:", len(multi))