"""Train extra siamese BiLSTM seeds from the command line (same code and settings as train.ipynb).

    python siamese/train_seeds.py mean 1 2     # pooling, then seeds

Seeds whose checkpoint already exists are skipped. Each finished seed is scored on val and test
(predictions/siamese_<pooling>/probs_{val,test}_seed_<s>.npy). Folders follow helpers/paths.py
(redirect with AM_DATA / AM_CHECKPOINTS / AM_PREDICTIONS).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                           # noqa: E402
import torch                                                                 # noqa: E402
from torch.utils.data import DataLoader                                     # noqa: E402

from helpers.pairs import load_pairs                                         # noqa: E402
from helpers.paths import probs_path                                         # noqa: E402
from siamese.model import build_vocab, PairDataset, make_collate             # noqa: E402
from siamese.training import (train_run, load_run, rebuild_model, predict, broken_run_problems,  # noqa: E402
                              checkpoint_path, history_path, run_name, CHECKPOINT_DIR)

DATASET = "overture_case2s"
BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE = 64, 1e-3, 20, 3     # as in train.ipynb

pooling, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train, train_label = load_pairs(DATASET, "train")
val, val_label = load_pairs(DATASET, "val")
vocab = build_vocab(train)
train_loader = DataLoader(PairDataset(train, train_label), batch_size=BATCH_SIZE, shuffle=True, collate_fn=make_collate(vocab))
val_loader = DataLoader(PairDataset(val, val_label), batch_size=BATCH_SIZE, shuffle=False, collate_fn=make_collate(vocab))

for seed in seeds:
    if not os.path.exists(checkpoint_path(seed, pooling)):
        history, best, best_state = train_run(seed, pooling, vocab, train_loader, val_loader, val, val_label,
                                              lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE)
        os.makedirs(os.path.dirname(history_path(seed, pooling)), exist_ok=True)
        history.to_csv(history_path(seed, pooling), index=False)
        torch.save({"state_dict": best_state, "vocab": vocab,
                    "config": {"emb_dim": 32, "hidden": 128, "pooling": pooling, "max_len": 85},
                    "seed": seed, "best": best}, checkpoint_path(seed, pooling))
    history, ckpt = load_run(seed, pooling)
    print(f"{pooling} seed {seed}: best {ckpt['best']} | problems {broken_run_problems(history)}", flush=True)
    model = rebuild_model(ckpt)
    for split in ["val", "test"]:
        df, labels = load_pairs(DATASET, split)
        loader = DataLoader(PairDataset(df, labels), batch_size=BATCH_SIZE, shuffle=False, collate_fn=make_collate(ckpt["vocab"]))
        np.save(probs_path(run_name(pooling), split, seed), predict(model, loader).numpy())
    print(f"{pooling} seed {seed}: val/test probabilities saved", flush=True)
