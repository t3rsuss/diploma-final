import sys
import csv
import hashlib
import pathlib

import numpy as np
import lightgbm as lgb
import lief

try:
    lief.logging.disable()
except Exception:
    pass

BLOCK = 512
HEADER, CODE, RODATA, DATA, RES, OTHER = range(6)
LIMIT = 30_000_000
HERE = pathlib.Path(__file__).resolve().parent


def enum_int(v):
    return int(getattr(v, "value", v))


def pe_class(s):
    name = s.name.lower()
    ch = int(s.characteristics)
    if name == ".rsrc":
        return RES
    if name in (".reloc", ".debug"):
        return OTHER
    if ch & 0x20000000 or ch & 0x20:
        return CODE
    if ch & 0x80000000:
        return DATA
    return RODATA


def elf_class(s):
    fl = int(s.flags)
    if fl & 4:
        return CODE
    if not fl & 2:
        return OTHER
    if fl & 1:
        return DATA
    return RODATA


def parse(path, size):
    b = lief.parse(str(path))
    if b is None:
        return None
    if isinstance(b, lief.PE.Binary):
        fmt, fn = "PE", pe_class
    elif isinstance(b, lief.ELF.Binary):
        fmt, fn = "ELF", elf_class
    else:
        return None
    lab = np.full(size, OTHER, dtype=np.uint8)
    spans, wx, end_max = [], 0, 0
    for s in b.sections:
        off, sz = int(s.offset), int(s.size)
        if fmt == "ELF" and enum_int(s.type) == 8:
            continue
        if sz <= 0 or off <= 0 or off >= size:
            continue
        end = min(off + sz, size)
        spans.append((off, end, fn(s)))
        if fmt == "PE":
            ch = int(s.characteristics)
            if (ch & 0x20000000) and (ch & 0x80000000):
                wx += 1
        else:
            fl = int(s.flags)
            if (fl & 4) and (fl & 1):
                wx += 1
        end_max = max(end_max, end)
    if not spans:
        return None
    first = min(a for a, _, _ in spans)
    lab[:first] = HEADER
    for a, e, c in spans:
        lab[a:e] = c
    return fmt, lab, len(spans), wx, end_max


def block_features(blocks):
    n = len(blocks)
    hist = np.zeros((n, 256), dtype=np.float32)
    for i in range(n):
        hist[i] = np.bincount(blocks[i], minlength=256)
    p = hist / BLOCK
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    ent = -(p * logp).sum(1)
    h64 = p.reshape(n, 64, 4).sum(2)
    extra = np.stack([
        ent, p[:, 0], p[:, 255], p[:, 32:127].sum(1), p[:, 128:].sum(1),
        blocks.mean(1) / 255.0, blocks.std(1) / 255.0, (hist > 0).sum(1) / 256.0,
    ], axis=1)
    return np.hstack([h64, extra]).astype(np.float32)


def structure_features(path, model):
    data = np.fromfile(str(path), dtype=np.uint8)
    n = len(data) // BLOCK
    if n < 2:
        return None
    res = parse(path, len(data))
    if res is None:
        return None
    fmt, lab, n_sec, wx, end_max = res
    blocks = data[: n * BLOCK].reshape(n, BLOCK)
    truth = lab[: n * BLOCK].reshape(n, BLOCK)[:, 0].astype(int)

    f = block_features(blocks)
    idx = np.arange(n)
    prev = f[np.clip(idx - 1, 0, n - 1)]
    nxt = f[np.clip(idx + 1, 0, n - 1)]
    rel = (idx / max(n - 1, 1)).astype(np.float32)[:, None]
    X = np.hstack([f, prev, nxt, rel])
    pred = np.asarray(model.predict(X)).argmax(axis=1)

    ent = f[:, 64]
    tc, pc = truth == CODE, pred == CODE
    ncode = max(int(tc.sum()), 1)
    out = {
        "n_blocks": n,
        "n_sections": n_sec,
        "wx_sections": wx,
        "overlay_ratio": max(len(data) - end_max, 0) / len(data),
        "agree": float((pred == truth).mean()),
        "code_outside_code_sections": float((pc & ~tc).mean()),
        "code_sections_not_code": float((tc & ~pc).sum() / ncode),
        "ent_mean": float(ent.mean()),
        "ent_max": float(ent.max()),
        "share_ent_gt7": float((ent > 7.0).mean()),
        "share_ent_gt65": float((ent > 6.5).mean()),
        "packed_code_share": float(((ent > 7.0) & tc).sum() / ncode),
    }
    for c in range(6):
        out[f"pred_share_{c}"] = float((pred == c).mean())
        out[f"table_share_{c}"] = float((truth == c).mean())
    return fmt, out, hashlib.sha256(data.tobytes()).hexdigest()


def main():
    folder = pathlib.Path(sys.argv[1])
    out_csv = pathlib.Path(sys.argv[2])
    model_path = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "block_clf.txt"
    model = lgb.Booster(model_file=str(model_path))
    files = [p for p in sorted(folder.rglob("*"))
             if p.is_file() and p.stat().st_size <= LIMIT]
    writer, fh, done = None, open(out_csv, "w", newline="", encoding="utf-8"), 0
    for i, p in enumerate(files, 1):
        try:
            r = structure_features(p, model)
        except Exception as e:
            print("пропущено", p.name, e)
            continue
        if r is None:
            continue
        fmt, feats, sha = r
        row = {"file": p.name, "sha256": sha, "kind": fmt, **feats}
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)
        done += 1
        if i % 50 == 0:
            print(f"{i}/{len(files)}", flush=True)
    fh.close()
    print("готово:", out_csv, "файлів:", done)


if __name__ == "__main__":
    main()