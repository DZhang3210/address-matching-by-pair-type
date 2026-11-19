"""Siamese BiLSTM training: the epoch loop and prediction, per-epoch evaluation, the early-stopping rule
shared by every neural model (paper, Experimental Setup), broken-run checks, one full run, and loading saved runs.

The model itself: model.py. Notebook: train.ipynb. The protocol (evaluate, is_better, THRESHOLD,
CASE1_GUARDRAIL, broken_run_problems) is also imported by esim/ and distilbert/.
"""
import os

import copy

import pandas as pd
import torch
from torch import nn

from helpers.check_model_performance import review_typical_slices

from helpers.paths import CHECKPOINTS, PREDICTIONS
from siamese.model import Encoder, SiameseMatcher

THRESHOLD = 0.5
CASE1_GUARDRAIL = 98.0      # % rejected on val case1; an epoch below it can't be "best"
COIN_FLIP_LOSS = 0.693      # -ln 0.5: a model that has learned nothing
BASE_RATE_LOSS = 0.325      # -(0.1 ln 0.1 + 0.9 ln 0.9): always predicting the 1:9 class mix
CHECKPOINT_DIR = os.path.join(CHECKPOINTS, "siamese")   # git-ignored


# ---------- epoch loop and prediction ----------

def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss, n_pairs = 0.0, 0
    for ids_a, len_a, ids_b, len_b, y in loader:
        opt.zero_grad()                                              # first: clear anything left from before
        logits = model(ids_a, len_a, ids_b, len_b)                   # the model never sees y
        loss = loss_fn(logits, y)                                    # BCEWithLogitsLoss applies the sigmoid itself
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # between backward and step: edits the gradients step uses
        opt.step()
        total_loss += loss.item() * len(y)                           # weight by batch size -> average per pair, not per batch
        n_pairs += len(y)
    return total_loss / n_pairs


def predict(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():                                            # no graph recorded, no updates
        for ids_a, len_a, ids_b, len_b, y in loader:                 # y unused
            logits = model(ids_a, len_a, ids_b, len_b)
            probs.append(torch.sigmoid(logits))                      # logits -> probabilities
    return torch.cat(probs)                                          # (N,), in loader order: val/test loaders must not shuffle


# ---------- evaluation, early stopping, broken-run checks ----------

def evaluate(model, loader, df, labels):
    """One model on one split -> {"val_loss", "found", "case1", "easy"} (% correct per slice)."""
    probs = predict(model, loader)
    y_true = torch.tensor(labels.values, dtype=torch.float32)
    preds = (probs >= THRESHOLD).int().numpy()
    val_loss = nn.functional.binary_cross_entropy(probs, y_true).item()
    table = review_typical_slices(df, preds, labels).set_index("slice")
    pct = lambda name: float(100 * table.loc[name, "correct"] / table.loc[name, "n"])
    found = pct("positives: non-identical")
    case1 = pct("neg: case1 (same street, diff number)")
    easy = pct("neg: case3 (easy)")

    return {"val_loss": val_loss, "found": found, "case1": case1, "easy": easy}


def is_better(row, best):
    """Early-stopping rule: below the case1 guardrail never qualifies; then higher found; tie -> lower val loss."""
    if row["case1"] < CASE1_GUARDRAIL:
        return False
    if best is None:
        return True
    return (row["found"] > best["found"]) or (row["found"] == best["found"] and row["val_loss"] < best["val_loss"])


def broken_run_problems(history):
    """One run's history -> list of problems; [] = healthy."""
    problems = []
    if history.isna().any().any():
        problems.append("NaN in history")
    first_val_loss = history["val_loss"].iloc[0]
    if first_val_loss >= COIN_FLIP_LOSS:
        problems.append(f"val loss after epoch 0 = {first_val_loss:.3f} (coin flip = {COIN_FLIP_LOSS})")
    min_train_loss = history["train_loss"].min()
    if min_train_loss >= 0.30:
        problems.append(f"train loss never below 0.30 (min {min_train_loss:.3f}; base rate = {BASE_RATE_LOSS})")
    max_found = history["found"].max()
    if max_found == 0:
        problems.append("val found = 0% in every epoch")
    return problems


# ---------- one full run ----------

def train_run(seed, pooling, vocab, train_loader, val_loader, val_df, val_labels, lr=1e-3, max_epochs=20, patience=3):
    """One training run -> (history DataFrame, best epoch's row, best epoch's state_dict).

    Early stopping on `is_better`; with no epoch above the case1 guardrail, the last epoch's weights are kept.
    """
    torch.manual_seed(seed)
    model = SiameseMatcher(Encoder(len(vocab), pooling=pooling))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best, best_state, epochs_since_best = None, None, 0
    history = []
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, opt, loss_fn)

        row = {"epoch": epoch, "train_loss": train_loss, **evaluate(model, val_loader, val_df, val_labels)}
        print(row)
        history.append(row)

        if is_better(row, best):
            best, best_state, epochs_since_best = row, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break
    if best is None:                                      # no epoch reached the case1 guardrail -> keep the last epoch's weights
        best_state = copy.deepcopy(model.state_dict())
    return pd.DataFrame(history), best, best_state


# ---------- reading saved runs ----------

def run_name(pooling):
    """Folder name of a siamese run under predictions/ (history + probabilities)."""
    return f"siamese_{pooling}"


def checkpoint_path(seed, pooling, checkpoint_dir=CHECKPOINT_DIR):
    return os.path.join(checkpoint_dir, f"overture_siamese_seed_{seed}_pooling_{pooling}.pt")


def history_path(seed, pooling):
    return os.path.join(PREDICTIONS, run_name(pooling), f"history_seed_{seed}.csv")


def load_run(seed, pooling, checkpoint_dir=CHECKPOINT_DIR):
    """One saved run -> (per-epoch history DataFrame, checkpoint dict with state_dict / vocab / config / seed / best)."""
    history = pd.read_csv(history_path(seed, pooling))
    ckpt = torch.load(checkpoint_path(seed, pooling, checkpoint_dir))
    return history, ckpt


def rebuild_model(ckpt):
    """Checkpoint -> the model that produced ckpt["best"], built from the checkpoint's own vocab + config."""
    model = SiameseMatcher(Encoder(len(ckpt["vocab"]), pooling=ckpt["config"]["pooling"]))
    model.load_state_dict(ckpt["state_dict"])
    return model
