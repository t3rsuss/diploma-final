import sys
import inspect
import pathlib
import thrember

ROOT = pathlib.Path(__file__).resolve().parents[1]

for name in ("download_models", "download_dataset", "predict_sample",
             "create_vectorized_features", "read_vectorized_features",
             "train_model"):
    f = getattr(thrember, name)
    print("=" * 60)
    print(name, inspect.signature(f))
    print((inspect.getdoc(f) or "без опису")[:700])

ext = thrember.PEFeatureExtractor()
print("=" * 60)
print("методи екстрактора:", [m for m in dir(ext) if not m.startswith("_")])

files = sorted((ROOT / "samples" / "clean").glob("*.exe"))
if files:
    p = files[0]
    data = p.read_bytes()
    print("тестовий файл:", p.name, len(data), "байт")
    for name in ("feature_vector", "raw_features"):
        if hasattr(ext, name):
            try:
                r = getattr(ext, name)(data)
                size = getattr(r, "shape", None) or (len(r) if hasattr(r, "__len__") else "")
                print(name, "->", type(r).__name__, size)
            except Exception as e:
                print(name, "помилка:", e)