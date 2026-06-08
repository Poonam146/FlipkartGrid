import pandas as pd

cat = pd.read_csv("submission.csv")
lgb = pd.read_csv("submission_lgbm.csv")

weights = [
    (0.45, 0.55),
    (0.42, 0.58),
    (0.38, 0.62),
    (0.35, 0.65)
]

for wc, wl in weights:
    sub = cat.copy()
    sub["demand"] = wc * cat["demand"] + wl * lgb["demand"]
    sub.to_csv(f"ensemble_{wc}_{wl}.csv", index=False)

print("done")