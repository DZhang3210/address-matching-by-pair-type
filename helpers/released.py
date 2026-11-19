"""Load the released weights (Hugging Face: zhangdav/address-matching-by-pair-type).

Each run is a folder <family>/<variant>/seed_<s>/ holding model.safetensors and config.json:
  siamese/{mean,max}/seed_0
  esim/{mc2,mc1,ft_mc1,rand_mc2,stack_mc1}/seed_{0,1,2}
  distilbert/{bi,cross}/seed_{0,1,2}

    from helpers.released import load_released
    model, collate = load_released("esim", "rand_mc2", seed=0)          # downloads the run
    loader = DataLoader(PairDataset(df, labels), batch_size=64, collate_fn=collate)

`collate` turns (address_a, address_b, label) samples into the model's inputs; the model returns logits
(apply a sigmoid for probabilities; the paper's threshold is 0.5).
"""
import json
import os

import torch
from safetensors.torch import load_file

HF_REPO = "zhangdav/address-matching-by-pair-type"


def run_folder(family, variant, seed, root=None):
    """Local folder of one run: under `root` if given, else downloaded from the Hugging Face repo."""
    sub = f"{family}/{variant}/seed_{seed}"
    if root is not None:
        return os.path.join(root, sub)
    from huggingface_hub import snapshot_download
    return os.path.join(snapshot_download(HF_REPO, allow_patterns=[f"{sub}/*"]), sub)


def load_released(family, variant, seed=0, root=None):
    """-> (model in eval mode, collate function)."""
    folder = run_folder(family, variant, seed, root)
    state = load_file(os.path.join(folder, "model.safetensors"))
    config = json.load(open(os.path.join(folder, "config.json")))

    if family == "siamese":
        from siamese.model import Encoder, SiameseMatcher, make_collate
        model = SiameseMatcher(Encoder(len(config["vocab"]), pooling=config["config"]["pooling"]))
        collate = make_collate(config["vocab"])
    elif family == "esim":
        from esim.model import ESIM, make_collate
        model = ESIM(torch.zeros_like(state["embedding.frozen.weight"]))   # the table itself is in the state dict
        collate = make_collate(config["vocab"])
    elif family == "distilbert":
        from distilbert.model import MODELS, make_bi_collate, make_cross_collate, get_tokenizer
        model = MODELS[variant]()
        collate = make_bi_collate(get_tokenizer().get_vocab()) if variant == "bi" else make_cross_collate()
    else:
        raise ValueError(f"family must be 'siamese', 'esim' or 'distilbert', not {family!r}")

    model.load_state_dict(state)
    return model.eval(), collate
