import os

import pandas as pd

# Both datasets share one pair schema:
#   id_a, id_b, address_a, address_b, zip_a, zip_b, split, label, neg_type (case0 = positive, case1-3 = negatives)
# Written by pipeline/02_build_pairs.ipynb ("overture") and pipeline/03_case2s.ipynb ("overture_case2s", the paper's dataset).

from helpers.paths import DATA

OVERTURE_RELEASE = "2026-09-23.1"


def _path(dataset, split, release):
    if dataset == "overture":
        return os.path.join(DATA, f"ct_pairs_{split}_{release}.parquet")
    if dataset == "overture_case2s":   # Overture pairs with look-alike street swaps replacing some easy negatives
        return os.path.join(DATA, f"ct_pairs_{split}_case2s.parquet")
    raise ValueError(f"Unknown dataset {dataset!r}: use 'overture' or 'overture_case2s'")


# One split -> (pairs without label, label). Works from any notebook folder.
def load_pairs(dataset, split, release=OVERTURE_RELEASE):
    df = pd.read_parquet(_path(dataset, split, release))
    return df.drop(columns=["label"]), df["label"]


# All three splits in one frame, label column re-attached (what the Overture swap notebooks start from).
def load_all_pairs(dataset, release=OVERTURE_RELEASE):
    frames = []
    for split in ["train", "val", "test"]:
        df, label = load_pairs(dataset, split, release)
        frames.append(df.assign(label=label))
    return pd.concat(frames, ignore_index=True)


# Split each address into house number + street remainder, for the similarity features.
# Number token = starts with a digit, runs to the first space (12, 12A, 16-18, 20-A).
# extract and replace use the same token so the number never stays on the street.
# No number -> NaN in number_* -> number_missing (one side) / both_number_missing (both sides).
# Returns a copy; the pairs frame itself is left unchanged.
def add_number_street(df):
    df = df.copy()
    for side in "ab":
        df[f"number_{side}"] = df[f"address_{side}"].str.extract(r"^(\d[A-Z0-9a-z-]*)\s", expand=False)
        df[f"street_{side}"] = df[f"address_{side}"].str.replace(r"^(\d[A-Z0-9a-z-]*\s+)", "", regex=True)
    df["both_number_missing"] = df["number_a"].isna() & df["number_b"].isna()
    df["number_missing"] = (df["number_a"].isna() | df["number_b"].isna()) & ~df["both_number_missing"]
    return df


# One hash per pair row from the columns every model reads (address + ZIP, both sides).
# The feature cache stores it and compare it on load, so a cache built from an
# older version of the pair files raises instead of being silently reused (2026-10-16).
# It does not see code changes: still delete the cache when features.py changes.
CACHE_KEY = "_pair_hash"
_FINGERPRINT_COLS = ["address_a", "address_b", "zip_a", "zip_b"]

def pair_fingerprint(df):
    return pd.util.hash_pandas_object(df[_FINGERPRINT_COLS], index=False).values


def check_cache(cached, df, path):
    if CACHE_KEY not in cached.columns or len(cached) != len(df) or (cached[CACHE_KEY].values != pair_fingerprint(df)).any():
        raise ValueError(f"Stale cache {path}: it doesn't match the current pair rows. Delete it and re-run.")
    return cached.drop(columns=[CACHE_KEY])
