import collections
import csv
import pathlib
import struct
import sys

MACH = {3: "x86", 4: "m68k", 8: "MIPS", 20: "PowerPC", 21: "PPC64", 40: "ARM",
        42: "SuperH", 62: "x86-64", 183: "AArch64", 243: "RISC-V"}

folder = pathlib.Path(sys.argv[1])
out_csv = sys.argv[2] if len(sys.argv) > 2 else None
cnt = collections.Counter()
rows = []
for p in sorted(folder.rglob("*")):
    if not p.is_file():
        continue
    with open(p, "rb") as f:
        h = f.read(20)
    if len(h) < 20 or h[:4] != b"\x7fELF":
        continue
    le = h[5] == 1
    m = struct.unpack("<H" if le else ">H", h[18:20])[0]
    arch = MACH.get(m, str(m))
    bits = {1: 32, 2: 64}.get(h[4], "?")
    endian = "LE" if le else "BE"
    rows.append((p.name, arch, bits, endian))
    cnt[(arch, bits, endian)] += 1

print("усього ELF:", len(rows))
for k, v in cnt.most_common():
    print(f"{v:5d}  {k[0]} {k[1]}-біт {k[2]}")

if out_csv:
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "arch", "bits", "endian"])
        w.writerows(rows)
    print("збережено:", out_csv)