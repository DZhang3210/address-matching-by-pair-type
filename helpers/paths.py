"""Repository paths, absolute, so every notebook and module finds the same folders whatever its working directory.

- DATA:        the pair files written by pipeline/ (git-ignored, rebuilt by the pipeline notebooks)
- CHECKPOINTS: trained models and word-vector tables (git-ignored, rebuilt by the training notebooks)
- PREDICTIONS: each neural run's val/test probabilities + training history (committed, so the paper's tables
               can be reproduced without retraining; results/paper_tables.ipynb reads them)
- FEATURES:    cached similarity features for the classical models (git-ignored)

Each can be redirected with an environment variable (AM_DATA, AM_CHECKPOINTS, AM_PREDICTIONS, AM_FEATURES),
e.g. to train on a rebuilt dataset without touching the saved runs.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.environ.get("AM_DATA", os.path.join(ROOT, "pipeline", "Data"))
CHECKPOINTS = os.environ.get("AM_CHECKPOINTS", os.path.join(ROOT, "checkpoints"))
PREDICTIONS = os.environ.get("AM_PREDICTIONS", os.path.join(ROOT, "predictions"))
FEATURES = os.environ.get("AM_FEATURES", os.path.join(ROOT, "classical", "features"))


def probs_path(run, split, seed):
    """Saved probabilities of one run: predictions/<run>/probs_<split>_seed_<seed>.npy."""
    return os.path.join(PREDICTIONS, run, f"probs_{split}_seed_{seed}.npy")
