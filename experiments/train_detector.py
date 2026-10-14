import sys
import json
import pathlib
import numpy as np
import joblib
import lightgbm as lgb
import thrember
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

ROOT = pathlib.Path(__file__).resolve().parents[1]
KIND = sys.argv[1] if len(sys.argv) > 1 else "elf"
PRETRAINED = {"elf": "ELF", "win64": "Win64"}[KIND]
MAX_FIT = 300_000
MAX_CAL = 50_000

d = ROOT / "data" / f"ember2024_{KIND}"
Xtr, ytr = thrember.read_vectorized_features(d, "train")
Xte, yte = thrember.read_vectorized_features(d, "test")

ok = np.where((ytr == 0) | (ytr == 1))[0]
cut = int(len(ok) * 0.9)
early, late = ok[:cut], ok[cut:]
rng = np.random.default_rng(42)
fit_idx = np.sort(rng.choice(early, min(MAX_FIT, len(early)), replace=False))
cal_idx = np.sort(rng.choice(late, min(MAX_CAL, len(late)), replace=False))
print("навчання:", len(fit_idx), "калібрування (останні 10 %):", len(cal_idx), flush=True)

dfit = lgb.Dataset(Xtr[fit_idx], label=ytr[fit_idx])
dcal = lgb.Dataset(Xtr[cal_idx], label=ytr[cal_idx], reference=dfit)
params = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=64, feature_fraction=0.5, bagging_fraction=0.8,
              bagging_freq=1, min_data_in_leaf=50, verbose=-1)
model = lgb.train(params, dfit, num_boost_round=2000, valid_sets=[dcal],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

p_cal = model.predict(Xtr[cal_idx])
y_cal = np.asarray(ytr[cal_idx])
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_cal)
neg = p_cal[y_cal == 0]
thr1 = float(np.quantile(neg, 0.99))
thr01 = float(np.quantile(neg, 0.999))

mask = (yte == 0) | (yte == 1)
Xt, yt = np.asarray(Xte[mask]), np.asarray(yte[mask])


def tpr_at(y, p, fpr_target):
    fpr, tpr, thr = roc_curve(y, p)
    i = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[max(i, 0)])


pretrained = lgb.Booster(model_file=str(ROOT / "data" / "ember_models"
                                        / f"EMBER2024_{PRETRAINED}.model"))
p_own = model.predict(Xt)
p_pre = pretrained.predict(Xt)
res = {}
for name, p in (("власна", p_own), ("готова EMBER2024", p_pre)):
    res[name] = {"AUC": float(roc_auc_score(yt, p)),
                 "TPR при FPR 1%": tpr_at(yt, p, 0.01),
                 "TPR при FPR 0,1%": tpr_at(yt, p, 0.001)}
    print(f"\n{name}\n  AUC {res[name]['AUC']:.4f}"
          f"\n  TPR при FPR 1 %: {res[name]['TPR при FPR 1%']:.4f}"
          f"\n  TPR при FPR 0,1 %: {res[name]['TPR при FPR 0,1%']:.4f}")

print("\nпороги, підібрані на контрольній вибірці, і результат на тесті:")
for label, thr in (("FPR 1 %", thr1), ("FPR 0,1 %", thr01)):
    real_fpr = float((p_own[yt == 0] > thr).mean())
    real_tpr = float((p_own[yt == 1] > thr).mean())
    res[f"поріг {label}"] = {"поріг": thr, "FPR на тесті": real_fpr, "TPR на тесті": real_tpr}
    print(f"  {label}: поріг {thr:.4f} -> FPR на тесті {real_fpr:.4f}, TPR {real_tpr:.4f}")

p_iso = iso.predict(p_own)
res["брайєр до калібрування"] = float(brier_score_loss(yt, p_own))
res["брайєр після калібрування"] = float(brier_score_loss(yt, p_iso))
print("\nпомилка Брайєра до/після калібрування:",
      round(res["брайєр до калібрування"], 5),
      round(res["брайєр після калібрування"], 5))

out = ROOT / "src" / "models"
out.mkdir(parents=True, exist_ok=True)
model.save_model(str(out / f"detector_{KIND}.txt"))
joblib.dump(iso, out / f"calibrator_{KIND}.joblib")
(out / f"thresholds_{KIND}.json").write_text(
    json.dumps({"thr_fpr_1": thr1, "thr_fpr_01": thr01}), encoding="utf-8")
(ROOT / "results" / f"detector_{KIND}_metrics.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nмодель, пороги і метрики збережено")