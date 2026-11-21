"""Refit the random forest and XGBoost over several random_state seeds and report the paper's test slices.

    python tools/classical_seeds.py 0 1 2 3 4

Same features and library defaults as classical/models.py, only random_state changes. Logistic regression
is deterministic (lbfgs), so it is not repeated. Prints per-seed rows and a min–max summary for the
sentinel and zero-fill settings. Folders follow helpers/paths.py (AM_DATA / AM_FEATURES).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from sklearn.ensemble import RandomForestClassifier                   # noqa: E402
from sklearn.metrics import roc_auc_score                             # noqa: E402
from xgboost import XGBClassifier                                     # noqa: E402

from helpers.pairs import load_pairs, add_number_street               # noqa: E402
from helpers.paths import FEATURES                                    # noqa: E402
from helpers.features import get_processed_dfs                        # noqa: E402

seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
P = {s: load_pairs("overture_case2s", s) for s in ["train", "val", "test"]}
df = {s: add_number_street(d) for s, (d, _) in P.items()}
y = {s: lab.values for s, (_, lab) in P.items()}
X = {}
X["sentinel"] = dict(zip(["train", "val", "test"], get_processed_dfs(
    [[df[s], f"{FEATURES}/{s}.parquet"] for s in ["train", "val", "test"]])))
X["zero-fill"] = dict(zip(["train", "val", "test"], get_processed_dfs(
    [[df[s], f"{FEATURES}/{s}_missing_as_zero.parquet"] for s in ["train", "val", "test"]], missing_number_as_zero=True)))

t, yt = df["test"], y["test"]
nonid = (yt == 1) & (t.address_a != t.address_b).values
c2s = (t.neg_type == "case2s").values
missing = nonid & (t.number_a.notna() != t.number_b.notna()).values
SLICES = {"Identical found": ((yt == 1) & (t.address_a == t.address_b).values, 1),
          "Non-identical found": (nonid, 1),
          "Different number rejected": ((t.neg_type == "case1").values, 0),
          "Look-alike rejected": (c2s, 0),
          "Random rejected": ((t.neg_type == "case3").values, 0)}
auc = lambda p, a: roc_auc_score(np.r_[np.ones(a.sum()), np.zeros(c2s.sum())], np.r_[p[a], p[c2s]])

rows = []
for setting, Xs in X.items():
    for name, make in [("RF", lambda s: RandomForestClassifier(random_state=s)), ("XGB", lambda s: XGBClassifier(random_state=s))]:
        for seed in seeds:
            m = make(seed).fit(Xs["train"], y["train"])
            p = m.predict_proba(Xs["test"])[:, 1]
            row = {"model": f"{name} {setting}", "seed": seed}
            row.update({k: 100 * ((p[mask] >= 0.5) == lab).mean() for k, (mask, lab) in SLICES.items()})
            row["AUC, non-identical vs look-alike"] = auc(p, nonid)
            row["AUC, missing-number vs look-alike"] = auc(p, missing)
            rows.append(row)
            print(f"{name} {setting} seed {seed} done", flush=True)

res = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print(res.round(3).to_string(index=False))
print("\nmin–max over seeds:")
print(res.drop(columns="seed").groupby("model").agg(lambda v: f"{v.min():.3f}–{v.max():.3f}").to_string())
