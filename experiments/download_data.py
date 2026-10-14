import pathlib
import thrember

ROOT = pathlib.Path(__file__).resolve().parents[1]

for ftype in ("ELF", "Win64"):
    d = ROOT / "data" / f"ember2024_{ftype.lower()}"
    d.mkdir(parents=True, exist_ok=True)
    thrember.download_dataset(d, split="train", file_type=ftype)
    thrember.download_dataset(d, split="test", file_type=ftype)
    print("готово:", ftype)