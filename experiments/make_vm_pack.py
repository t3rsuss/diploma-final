import json
import shutil
import pathlib
import joblib

ROOT = pathlib.Path(__file__).resolve().parents[1]
pack = ROOT / "vm_pack"
if pack.exists():
    shutil.rmtree(pack)
pack.mkdir()

M = ROOT / "src" / "models"
for key in ("elf", "win64"):
    shutil.copy(M / f"detector_{key}.txt", pack)
    shutil.copy(M / f"thresholds_{key}.json", pack)
    iso = joblib.load(M / f"calibrator_{key}.joblib")
    (pack / f"calibrator_{key}.json").write_text(
        json.dumps({"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}),
        encoding="utf-8")
for name in ("ELF", "Win64", "Win32", "Dot_Net", "PE"):
    shutil.copy(ROOT / "data" / "ember_models" / f"EMBER2024_{name}.model", pack)
shutil.copy(ROOT / "experiments" / "vm_score.py", pack)
shutil.make_archive(str(ROOT / "vm_pack"), "zip", pack)
print("архів створено:", ROOT / "vm_pack.zip")