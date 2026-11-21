"""Train the DistilBERT bi- or cross-encoder from the command line (same code as train_bi / train_cross.ipynb).

    python distilbert/train.py bi            # or: cross
    python distilbert/train.py bi --seeds 0  # a subset of seeds

Seeds whose seed_{s}_final.pt already exists are skipped, and an interrupted seed resumes from its last
finished epoch. Afterwards every finished seed is scored on val and test (predictions/distilbert_<kind>/).
Folders come from helpers/paths.py (redirect with AM_DATA / AM_CHECKPOINTS / AM_PREDICTIONS).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distilbert.model import make_loaders                                   # noqa: E402
from distilbert.training import run_training, save_predictions, checkpoint_dir, SEEDS, DEVICE   # noqa: E402
from helpers.pairs import load_pairs                                        # noqa: E402
from siamese.training import broken_run_problems                            # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=["bi", "cross"])
parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
args = parser.parse_args()

splits = {s: load_pairs("overture_case2s", s) for s in ["train", "val", "test"]}
loaders = make_loaders(args.kind, splits)
val, val_label = splits["val"]
print(f"{args.kind}: device {DEVICE}, checkpoints in {checkpoint_dir(args.kind)}", flush=True)

for seed in args.seeds:
    if os.path.exists(f"{checkpoint_dir(args.kind)}/seed_{seed}_final.pt"):
        print(f"seed {seed}: already finished, skipping", flush=True)
        continue
    history, best, _ = run_training(args.kind, seed, loaders, val, val_label)
    print(f"seed {seed}: best {best} | problems {broken_run_problems(history)}", flush=True)

for seed in args.seeds:
    save_predictions(args.kind, seed, loaders)
    print(f"seed {seed}: val/test probabilities saved", flush=True)
