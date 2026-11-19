"""ESIM runs: train and save one run, then read saved runs back.

Protocol (shared with the siamese BiLSTM): BCE, Adam 1e-3, best epoch by is_better (case1 guardrail 98),
patience 3, 3 seeds. The model itself: model.py. Notebook: 02_train.ipynb.

Where a run lives: its checkpoint in checkpoints/esim/<run>/ (git-ignored), its history and val/test
probabilities in predictions/<run>/ (committed). Every reader below works from predictions/ only;
a prediction is probability >= THRESHOLD.
"""
import copy
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from gensim.models import Word2Vec

from esim.model import ESIM, CHECKPOINTS, tokenize
from helpers.check_model_performance import review_typical_slices
from helpers.paths import PREDICTIONS, probs_path
from siamese.training import predict, train_epoch
from siamese.training import evaluate, is_better, broken_run_problems, THRESHOLD

LR, MAX_EPOCHS, PATIENCE = 1e-3, 20, 3
SEEDS = [0, 1, 2]

# setting name -> run name (folder under checkpoints/esim/ and predictions/)
SETTINGS = {
    "mc2":       "esim_mc2",        # word2vec min_count=2 (the reproduction)
    "mc1":       "esim_mc1",        # word2vec min_count=1
    "ft_mc1":    "esim_ft_mc1",     # fastText min_count=1
    "rand_mc2":  "esim_rand_mc2",   # word2vec mc2 + a fixed random row per dropped token (the kept setting)
    "stack_mc1": "esim_stack_mc1",  # [word2vec mc1; fastText mc1]
}
# vocab that defines the rare-number group
W2V_PATH = os.path.join(CHECKPOINTS, "word2vec_50_mc2.model")


# ---------- train + save ----------

def run_training(matrix, loaders, val, val_label, seed):
    """Seeded run with early stopping on is_better -> (history DataFrame, best row, best state_dict)."""
    torch.manual_seed(seed)
    model = ESIM(matrix)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, since_best, history = None, None, 0, []
    for epoch in range(MAX_EPOCHS):
        row = {"epoch": epoch, "train_loss": train_epoch(model, loaders["train"], opt, loss_fn),
               **evaluate(model, loaders["val"], val, val_label)}
        print(row)
        history.append(row)
        if is_better(row, best):
            best, best_state, since_best = row, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break
    if best is None:                                  # no epoch reached the case1 guardrail -> keep the last
        best_state = copy.deepcopy(model.state_dict())
    return pd.DataFrame(history), best, best_state


def save_run(run, seed, history, best, best_state, vocab, matrix, loaders, config):
    """checkpoints/esim/<run>/esim_seed_{s}.pt; predictions/<run>/history_seed_{s}.csv + probs_{val,test}_seed_{s}.npy."""
    ckpt_dir, pred_dir = os.path.join(CHECKPOINTS, run), os.path.join(PREDICTIONS, run)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    history.to_csv(os.path.join(pred_dir, f"history_seed_{seed}.csv"), index=False)
    torch.save({"state_dict": best_state, "vocab": vocab, "config": config, "seed": seed, "best": best},
               os.path.join(ckpt_dir, f"esim_seed_{seed}.pt"))
    model = ESIM(matrix)
    model.load_state_dict(best_state)
    for split in ["val", "test"]:
        np.save(probs_path(run, split, seed), predict(model, loaders[split]).numpy())


# ---------- read saved runs (predictions/ only) ----------

def load_preds(run, split, seed):
    """Thresholded predictions of one saved run."""
    return (np.load(probs_path(run, split, seed)) >= THRESHOLD).astype(int)


def _available(runs):
    """(name, run, seed) for every run whose val probabilities exist. runs: {display name: run name}."""
    return [(name, run, seed) for name, run in runs.items() for seed in SEEDS
            if os.path.exists(probs_path(run, "val", seed))]


def health(runs=SETTINGS):
    """One row per run: epochs trained, broken_run_problems ([] = healthy)."""
    rows = []
    for name, run, seed in _available(runs):
        h = pd.read_csv(os.path.join(PREDICTIONS, run, f"history_seed_{seed}.csv"))
        rows.append({"setting": name, "seed": seed, "epochs": len(h), "problems": broken_run_problems(h)})
    return pd.DataFrame(rows).set_index(["setting", "seed"])


def slice_table(split_df, labels, split, runs=SETTINGS):
    """% correct per slice (rows) for every (setting, seed) (columns)."""
    cols = {(name, seed): review_typical_slices(split_df, load_preds(run, split, seed), labels)
                          .set_index("slice")["% correct"]
            for name, run, seed in _available(runs)}
    return pd.DataFrame(cols)


def rare_number_group(split_df, group_vocab=None, w2v_path=W2V_PATH):
    """The rare-number group: both addresses start with a number unknown to the word2vec mc2 vocab,
    and the rest of the address is identical. Fixed vocab, so every setting is compared on the same rows."""
    if group_vocab is None:
        group_vocab = Word2Vec.load(w2v_path).wv.key_to_index
    strip_number = lambda s: s.str.replace(r"^(\d+\s+)", "", regex=True)
    grp = np.ones(len(split_df), dtype=bool)
    for side in ["a", "b"]:
        addr = split_df[f"address_{side}"]
        first_unknown = addr.apply(lambda x: (tokenize(x) or [None])[0] not in group_vocab)
        grp &= first_unknown.values & addr.str.contains(r"^\d+").values
    return grp & (strip_number(split_df["address_a"]) == strip_number(split_df["address_b"])).values


def rare_number_table(splits, runs=SETTINGS, w2v_path=W2V_PATH):
    """splits: {"val": (df, labels), "test": ...}. Per (split, setting, seed): identical matches found and
    case1 non-matches rejected inside the rare-number group, and errors outside it."""
    group_vocab = Word2Vec.load(w2v_path).wv.key_to_index
    rows = []
    for split, (split_df, labels) in splits.items():
        grp = rare_number_group(split_df, group_vocab)
        ident = (split_df["address_a"] == split_df["address_b"]).values & (labels == 1).values
        case1 = (split_df["neg_type"] == "case1").values & (labels == 0).values
        for name, run, seed in _available(runs):
            p = load_preds(run, split, seed)
            rows.append({
                "split": split, "setting": name, "seed": seed,
                "identical, rare: found":    f"{(p[ident & grp] == 1).sum()} / {(ident & grp).sum()}",
                "case1, rare: rejected":     f"{(p[case1 & grp] == 0).sum()} / {(case1 & grp).sum()}",
                "identical, other: missed":  int((p[ident & ~grp] == 0).sum()),
                "case1, other: errors":      int((p[case1 & ~grp] == 1).sum()),
            })
    return pd.DataFrame(rows).set_index(["split", "setting", "seed"])
