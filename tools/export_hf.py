"""Stage the Hugging Face model and dataset repos from the local checkpoints, predictions and pair files.

    python tools/export_hf.py <out_dir>

Writes <out_dir>/model/ (one folder per run: <family>/<variant>/seed_<s>/model.safetensors + config.json,
plus predictions/) and <out_dir>/dataset/ (case2s/ and original/ splits). helpers/released.py reads the
model layout. Uploading is a separate step (huggingface_hub.upload_folder); the cards (README.md) and the
dataset's licence files are not written here.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                              # noqa: E402
import torch                                                     # noqa: E402
from safetensors.torch import save_file                          # noqa: E402

from helpers.paths import CHECKPOINTS, DATA, PREDICTIONS          # noqa: E402

SEEDS = [0, 1, 2]
ESIM_SETTINGS = ["mc2", "mc1", "ft_mc1", "rand_mc2", "stack_mc1"]
COLUMNS = ["id_a", "id_b", "address_a", "address_b", "zip_a", "zip_b", "split", "neg_type", "label"]


def write_run(folder, state, config):
    os.makedirs(folder, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(folder, "model.safetensors"))
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=1)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def export_models(out):
    for pooling in ["mean", "max"]:
        for s in SEEDS:
            c = load(os.path.join(CHECKPOINTS, "siamese", f"overture_siamese_seed_{s}_pooling_{pooling}.pt"))
            write_run(os.path.join(out, "siamese", pooling, f"seed_{s}"), c["state_dict"],
                      {"family": "siamese", "pooling": pooling, "seed": s, "config": c["config"], "vocab": c["vocab"],
                       "best_epoch": c["best"]})
    for setting in ESIM_SETTINGS:
        for s in SEEDS:
            c = load(os.path.join(CHECKPOINTS, "esim", f"esim_{setting}", f"esim_seed_{s}.pt"))
            write_run(os.path.join(out, "esim", setting, f"seed_{s}"), c["state_dict"],
                      {"family": "esim", "setting": setting, "seed": s, "config": c["config"], "vocab": c["vocab"],
                       "best_epoch": c["best"]})
    for kind in ["bi", "cross"]:
        for s in SEEDS:
            c = load(os.path.join(CHECKPOINTS, f"distilbert_{kind}", f"seed_{s}_final.pt"))
            write_run(os.path.join(out, "distilbert", kind, f"seed_{s}"), c["best_state"],
                      {"family": "distilbert", "kind": kind, "seed": s, "base_model": "distilbert-base-uncased",
                       "best_epoch": c["best"]})
    shutil.copytree(PREDICTIONS, os.path.join(out, "predictions"), dirs_exist_ok=True)


def export_dataset(out):
    for config, name in [("case2s", "ct_pairs_{}_case2s.parquet"), ("original", "ct_pairs_{}_2026-09-23.1.parquet")]:
        os.makedirs(os.path.join(out, config), exist_ok=True)
        for split in ["train", "val", "test"]:   # same column order in both configs
            pd.read_parquet(os.path.join(DATA, name.format(split)))[COLUMNS].to_parquet(
                os.path.join(out, config, f"{split}.parquet"), index=False)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    export_models(os.path.join(out_dir, "model"))
    export_dataset(os.path.join(out_dir, "dataset"))
    print("staged in", out_dir)
