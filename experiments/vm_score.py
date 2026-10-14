import io
import sys
import csv
import json
import hashlib
import pathlib

import numpy as np
import lightgbm as lgb
import pefile
import thrember
import thrember.features as tf
from signify.authenticode import SignedPEFile

HERE = pathlib.Path(__file__).resolve().parent
LIMIT = 30_000_000
PRE_NAMES = {"elf": "ELF", "win64": "Win64", "dotnet": "Dot_Net",
             "win32": "Win32", "pe_generic": "PE"}


def _raw_features(self, bytez, pe):
    if pe is None:
        return {}
    raw_obj = {
        "num_certs": 0, "self_signed": 0, "empty_program_name": 0,
        "no_countersigner": 0, "parse_error": 0, "chain_max_depth": 0,
        "latest_signing_time": 0, "signing_time_diff": 0,
    }
    try:
        signed_pe = SignedPEFile(io.BytesIO(bytez))
        for signed_data in signed_pe.iter_signed_datas():
            raw_obj["num_certs"] += 1
            if signed_data.signer_info.program_name is None:
                raw_obj["empty_program_name"] = 1
            countersigner = signed_data.signer_info.countersigner
            if countersigner is not None:
                signing_time = countersigner.signing_time.timestamp()
                if signing_time >= raw_obj["latest_signing_time"]:
                    raw_obj["latest_signing_time"] = signing_time
                raw_obj["signing_time_diff"] = signing_time - pe.FILE_HEADER.TimeDateStamp
            else:
                raw_obj["no_countersigner"] = 1
            certs = list(signed_data.certificates)
            if len(certs) > raw_obj["chain_max_depth"]:
                raw_obj["chain_max_depth"] = len(certs)
            for cert in certs[:-1]:
                if cert.issuer == cert.subject:
                    raw_obj["self_signed"] = 1
    except Exception:
        raw_obj["parse_error"] = 1
    return raw_obj


tf.AuthenticodeSignature.raw_features = _raw_features
_CACHE = {}
_EXT = []


def extractor():
    if not _EXT:
        _EXT.append(thrember.PEFeatureExtractor())
    return _EXT[0]


def file_kind(path):
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] == b"MZ":
        return "PE"
    if head == b"\x7fELF":
        return "ELF"
    return None


def route(path):
    kind = file_kind(path)
    if kind is None:
        return None, None
    if kind == "ELF":
        return kind, "elf"
    try:
        pe = pefile.PE(str(path), fast_load=True)
        machine = pe.FILE_HEADER.Machine
        dirs = pe.OPTIONAL_HEADER.DATA_DIRECTORY
        dotnet = len(dirs) > 14 and dirs[14].VirtualAddress != 0
        pe.close()
    except Exception:
        return kind, "pe_generic"
    if dotnet:
        return kind, "dotnet"
    if machine == 0x8664:
        return kind, "win64"
    if machine == 0x14C:
        return kind, "win32"
    return kind, "pe_generic"


def pre_model(name):
    key = "pre_" + name
    if key not in _CACHE:
        _CACHE[key] = lgb.Booster(model_file=str(HERE / f"EMBER2024_{name}.model"))
    return _CACHE[key]


def own_model(key):
    if key not in _CACHE:
        _CACHE[key] = dict(
            b=lgb.Booster(model_file=str(HERE / f"detector_{key}.txt")),
            thr=json.loads((HERE / f"thresholds_{key}.json").read_text(encoding="utf-8")),
            cal=json.loads((HERE / f"calibrator_{key}.json").read_text(encoding="utf-8")),
        )
    return _CACHE[key]


def score(path):
    kind, key = route(path)
    if kind is None:
        return None
    with open(path, "rb") as f:
        data = f.read()
    vec = np.asarray(extractor().feature_vector(data), dtype=np.float32).reshape(1, -1)
    pre = float(pre_model(PRE_NAMES[key]).predict(vec)[0])
    if key in ("elf", "win64"):
        m = own_model(key)
        s = float(m["b"].predict(vec)[0])
        prob = float(np.interp(s, m["cal"]["x"], m["cal"]["y"]))
        t_mid, t_high = m["thr"]["thr_fpr_1"], m["thr"]["thr_fpr_01"]
    else:
        s, prob, t_mid, t_high = pre, "", 0.2, 0.5
    lvl = "high" if s >= t_high else ("mid" if s >= t_mid else "low")
    return kind, key, s, pre, prob, lvl, hashlib.sha256(data).hexdigest()


def main():
    folder = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    files = [p for p in sorted(folder.rglob("*"))
             if p.is_file() and p.stat().st_size <= LIMIT]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "sha256", "kind", "model", "score", "score_pre", "prob", "level"])
        for i, p in enumerate(files, 1):
            try:
                r = score(p)
            except Exception as e:
                print("пропущено", p.name, e)
                continue
            if r is None:
                continue
            kind, key, s, pre, prob, lvl, sha = r
            w.writerow([p.name, sha, kind, key, f"{s:.6f}", f"{pre:.6f}", prob, lvl])
            if i % 20 == 0:
                print(f"{i}/{len(files)}", flush=True)
    print("готово:", out)


if __name__ == "__main__":
    main()