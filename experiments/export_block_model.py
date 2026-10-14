import pathlib
import joblib

ROOT = pathlib.Path(__file__).resolve().parents[1]
d = joblib.load(ROOT / "src" / "models" / "block_clf.joblib")
out = ROOT / "src" / "models" / "block_clf.txt"
d["model"].booster_.save_model(str(out))
print("збережено:", out)