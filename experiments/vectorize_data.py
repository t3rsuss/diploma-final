import multiprocessing
import pathlib
import thrember

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    for ftype in ("elf", "win64"):
        d = ROOT / "data" / f"ember2024_{ftype}"
        ready = (d / "X_train.dat").exists() and (d / "X_test.dat").exists()
        if ready:
            print("уже готово:", ftype, flush=True)
        else:
            print("векторизація:", d, flush=True)
            try:
                thrember.create_vectorized_features(d)
            except ValueError as e:
                if "Did not find any .jsonl" not in str(e):
                    raise
                print("challenge-набору нема, пропускаємо", flush=True)
        for subset in ("train", "test"):
            X, y = thrember.read_vectorized_features(d, subset)
            print(ftype, subset, "форма", X.shape, "шкідливих", int(y.sum()), flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
    