import inspect
import pathlib
import thrember

ROOT = pathlib.Path(__file__).resolve().parents[1]

for name in ("download_dataset", "download_models"):
    print("=" * 60)
    print(name)
    try:
        print(inspect.getsource(getattr(thrember, name)))
    except Exception as e:
        print("код недоступний:", e)

out = ROOT / "data" / "ember_models"
out.mkdir(parents=True, exist_ok=True)
thrember.download_models(out)

print("=" * 60)
print("завантажені файли:")
for p in sorted(out.rglob("*")):
    if p.is_file():
        print(p.relative_to(out), round(p.stat().st_size / 1e6, 1), "МБ")