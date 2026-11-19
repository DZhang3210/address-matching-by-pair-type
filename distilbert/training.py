"""DistilBERT training: one run per seed, resumable after a disconnect, with the shared early-stopping rule
(siamese.training.is_better: case1 guardrail 98 on val, then most non-identical matches found, ties -> lower
val loss). AdamW (encoder 2e-5, head 1e-3, weight decay 0.01), linear warm-up over the first 10% of steps then
decay, at most 5 epochs, patience 2.

Where a run lives: checkpoints/distilbert_<kind>/seed_{s}_final.pt (git-ignored); history and val/test
probabilities in predictions/distilbert_<kind>/ (committed).
"""
import copy
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from distilbert.model import MODELS
from helpers.check_model_performance import review_typical_slices
from helpers.paths import CHECKPOINTS, PREDICTIONS, probs_path
from siamese.training import is_better, THRESHOLD

MAX_EPOCHS, PATIENCE = 5, 2
SEEDS = [0, 1, 2]
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")


def run_name(kind):
    return f"distilbert_{kind}"


def checkpoint_dir(kind):
    return os.path.join(CHECKPOINTS, run_name(kind))


# ---------- epoch loop and prediction (batch = (*model inputs, y)) ----------

def to_device(batch):
    return (t.to(DEVICE) for t in batch)


def train_epoch(model, loader, opt, sched, loss_fn):
    model.train()
    total_loss, n_pairs = 0.0, 0
    for batch in loader:
        *inputs, y = to_device(batch)
        opt.zero_grad()
        logits = model(*inputs)                                      # the model never sees y
        loss = loss_fn(logits, y)                                    # BCEWithLogitsLoss applies the sigmoid itself
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()                                                 # after opt.step(): sets the lr for the next batch
        total_loss += loss.item() * len(y)                           # average per pair, not per batch
        n_pairs += len(y)
    return total_loss / n_pairs


def predict(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            *inputs, y = to_device(batch)                            # y unused
            probs.append(torch.sigmoid(model(*inputs)).cpu())
    return torch.cat(probs)                                          # (N,), in loader order: val/test must not shuffle


def evaluate(model, loader, df, labels):
    """siamese.training.evaluate with this module's device-aware predict."""
    probs = predict(model, loader)
    y_true = torch.tensor(labels.values, dtype=torch.float32)
    preds = (probs >= THRESHOLD).int().numpy()
    val_loss = nn.functional.binary_cross_entropy(probs, y_true).item()
    table = review_typical_slices(df, preds, labels).set_index("slice")
    pct = lambda name: float(100 * table.loc[name, "correct"] / table.loc[name, "n"])
    return {"val_loss": val_loss,
            "found": pct("positives: non-identical"),
            "case1": pct("neg: case1 (same street, diff number)"),
            "easy": pct("neg: case3 (easy)")}


# ---------- one run ----------

def run_training(kind, seed, loaders, val, val_label, max_epochs=MAX_EPOCHS):
    """One seeded run -> (history DataFrame, best row, best state_dict). Saves seed_{s}_last.pt after every epoch
    (deleted at the end) so a disconnected run resumes, and seed_{s}_final.pt with the best epoch's weights."""
    torch.manual_seed(seed)
    ckpt_dir = checkpoint_dir(kind)
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = f"{ckpt_dir}/seed_{seed}_last.pt"
    final_path = f"{ckpt_dir}/seed_{seed}_final.pt"
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = MODELS[kind]().to(DEVICE)
    model.train()
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 2e-5},
        {"params": model.head.parameters(),    "lr": 1e-3},
    ], weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    best, best_state, history = None, None, []
    epochs_since_best, start_epoch = 0, 0
    if os.path.exists(last_path):                      # a disconnected run: continue after its last finished epoch
        ck = torch.load(last_path, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        history, best, best_state, epochs_since_best = ck["history"], ck["best"], ck["best_state"], ck["epochs_since_best"]
        start_epoch = ck["epoch"] + 1
        print(f"seed {seed}: resuming at epoch {start_epoch}")

    for epoch in range(start_epoch, max_epochs):
        train_loss = train_epoch(model, train_loader, opt, sched, loss_fn)
        row = {"epoch": epoch, "train_loss": train_loss, "lr_encoder": sched.get_last_lr()[0],
               **evaluate(model, val_loader, val, val_label)}
        print(row)
        history.append(row)
        if is_better(row, best):
            best, best_state, epochs_since_best = row, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                break
        state = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "history": history, "best": best, "best_state": best_state, "epochs_since_best": epochs_since_best}
        torch.save(state, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

    if best is None:                                   # no epoch reached the case1 guardrail -> keep the last epoch
        best_state = copy.deepcopy(model.state_dict())
    torch.save({"seed": seed, "best_state": best_state, "best": best, "history": history}, final_path)
    if os.path.exists(last_path):
        os.remove(last_path)
    return pd.DataFrame(history), best, best_state


def save_predictions(kind, seed, loaders):
    """Best epoch's weights -> predictions/distilbert_<kind>/probs_{val,test}_seed_{s}.npy + history_seed_{s}.csv."""
    ck = torch.load(f"{checkpoint_dir(kind)}/seed_{seed}_final.pt", map_location=DEVICE, weights_only=False)
    os.makedirs(os.path.join(PREDICTIONS, run_name(kind)), exist_ok=True)
    pd.DataFrame(ck["history"]).to_csv(os.path.join(PREDICTIONS, run_name(kind), f"history_seed_{seed}.csv"), index=False)
    model = MODELS[kind]().to(DEVICE)
    model.load_state_dict(ck["best_state"])
    for split in ["val", "test"]:
        np.save(probs_path(run_name(kind), split, seed), predict(model, loaders[split]).numpy())
