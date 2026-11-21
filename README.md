# Address matching by pair type: classical models vs. neural matchers

Code and saved model outputs for the paper [*Address Matching by Pair Type: Segmented Classical Models versus Raw-String Neural Matchers*](paper/address-matching-by-pair-type.pdf) (PDF in `paper/`). Trained weights: [zhangdav/address-matching-by-pair-type](https://huggingface.co/zhangdav/address-matching-by-pair-type) (models); the pairs: [datasets/zhangdav/address-matching-by-pair-type](https://huggingface.co/datasets/zhangdav/address-matching-by-pair-type). Given two address strings, each model predicts whether they refer to the same location. We compare classical models on hand-segmented addresses (logistic regression, random forest and XGBoost on 23 string-similarity features) with four neural models that read the raw strings (a character-level siamese BiLSTM, ESIM, and DistilBERT bi- and cross-encoders). All of them are trained and tested on the same splits of real cross-source address pairs from Connecticut, built from Overture Maps Places and Foursquare Open Source Places, and every model is scored separately on each pair type. The repository holds only what the paper uses: the data pipeline, the model code, and one notebook that reproduces every table and figure.

## Results

The kept model of each family on the test set (paper, Table 15). Percent of each pair type classified correctly at a 0.5 threshold; the last column is the ROC AUC of non-identical matches against look-alike streets. Neural models: min–max over three seeds.

| Model | Identical found | Non-identical found | Different number rejected | Look-alike rejected | Random rejected | AUC, non-identical vs look-alike |
|---|---|---|---|---|---|---|
| *n (pairs)* | *3,543* | *843* | *14,379* | *1,053* | *24,019* | |
| XGBoost (sentinel) | 100 | 83.6 | 100.0 | 87.0 | 99.9 | 0.920 |
| Siamese BiLSTM, mean pooling | 100 | 87.0–89.3 | 99.9–100.0 | 69.8–73.1 | 99.5–99.8 | 0.878–0.893 |
| ESIM, fastText | 100 | 87.3–90.3 | 98.1–99.0 | 73.5–80.4 | 99.8–99.9 | 0.861–0.890 |
| DistilBERT bi-encoder | 100 | 89.0–90.9 | 99.8–99.9 | 57.6–68.8 | 99.8–99.9 | 0.883–0.893 |
| DistilBERT cross-encoder | 100 | 80.3–83.0 | 99.9–100.0 | 86.1–88.8 | 99.9 | 0.888–0.904 |

The models differ most on matches where one side has no house number (164 test pairs): the random forest and XGBoost find 94–96% of them, the neural models 63–76%. With those matches set aside, every neural model ranks non-identical matches against look-alike streets about as well as XGBoost (paper, Table 13).

## Setup

Python 3.9 (the paper's runs used 3.9.6).

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`postal` (the Python binding for libpostal) needs the libpostal C library installed first; see the [libpostal README](https://github.com/openvenues/libpostal#installation-maclinux). Only the data pipeline needs it (`pipeline/03_case2s.ipynb` and `04_case2s_checks.ipynb` use it to flag two spellings of the same street).

Run every notebook from its own folder (each one adds the repository root to `sys.path`). Folders are set in `helpers/paths.py`.

## Using the trained models

All 27 neural runs are on Hugging Face as safetensors; `helpers/released.py` downloads and builds one:

```python
from helpers.released import load_released
model, collate = load_released("distilbert", "cross", seed=0)   # or ("esim", "ft_mc1", s), ("siamese", "mean", s), ...
batch = collate([("12 ELM ST", "12 ELM STREET", 1)])
probability = torch.sigmoid(model(*batch[:-1]))
```

## Reproducing the tables (no retraining)

The val/test probabilities of every neural run are committed in `predictions/`, so the tables need only the pair files:

1. Get the pairs: download the [dataset](https://huggingface.co/datasets/zhangdav/address-matching-by-pair-type) into `pipeline/Data/` as `ct_pairs_{train,val,test}_case2s.parquet`, or rebuild them with `pipeline/01` → `04` (below).
2. Run `results/paper_tables.ipynb`. It refits the three classical models from the similarity features (deterministic; the first run builds the feature cache in a few minutes), reads every neural run from `predictions/`, and writes Figures 1 and 3 to `results/figures/`. Table B1's earlier-version column is read from `results/v1_missing_number_matches.csv`.

## Run order (full rebuild)

| Step | Code | Writes (git-ignored unless noted) |
|---|---|---|
| 1 | `pipeline/01_overture_fsq.ipynb`: Overture places in CT, bridge files, Foursquare records, join, address and ZIP cleaning (DuckDB) | `pipeline/Data/ct_positives_*` |
| 2 | `pipeline/02_build_pairs.ipynb`: ZIP-based split, different-number, real look-alike and random pairs, 1:9 ratio | `pipeline/Data/ct_pairs_{split}_2026-09-23.1.parquet` |
| 3 | `pipeline/03_case2s.ipynb`: look-alike street swaps replace some random pairs (**the paper's dataset**) | `pipeline/Data/ct_pairs_{split}_case2s.parquet` |
| 4 | `pipeline/04_case2s_checks.ipynb`: acceptance checks on the look-alike street files | none |
| 5 | `siamese/train.ipynb` (seed 0), then `python siamese/train_seeds.py mean 1 2` and `python siamese/train_seeds.py max 1 2` | `checkpoints/siamese/`, `predictions/siamese_*` (committed) |
| 6 | `esim/01_word2vec.ipynb`: word2vec and fastText tables | `checkpoints/esim/*.model` |
| 7 | `esim/02_train.ipynb`: ESIM, five embedding settings × 3 seeds, plus the rare-number group masks | `checkpoints/esim/`, `predictions/esim_*` and `rare_number_group_*.npy` (committed) |
| 8 | `distilbert/train_bi.ipynb`, `distilbert/train_cross.ipynb`, or `python distilbert/train.py bi` / `cross`: 3 seeds each | `checkpoints/distilbert_*`, `predictions/distilbert_*` (committed) |
| 9 | `results/paper_tables.ipynb`: every table and figure | `results/figures/` (committed) |

**Seed scripts.** `siamese/train_seeds.py` trains further siamese seeds with the same code and settings as `siamese/train.ipynb`; `distilbert/train.py` trains the DistilBERT seeds from the command line (it also resumes an interrupted seed). Both skip seeds whose checkpoint exists and save the val/test probabilities. `tools/classical_seeds.py` refits the random forest and XGBoost over several `random_state` seeds (`python tools/classical_seeds.py 0 1 2 3 4`): XGBoost with its default settings gives identical results on every seed; the random forest's AUC varies within 0.921–0.924, and the paper reports seed 0. `tools/run_notebooks.py` executes notebooks in order without changing them and saves the executed copies in `checkpoints/logs/`.

**Data releases.** Overture Maps Places and bridge files, release `2026-09-23.1` (`OVERTURE_RELEASE`, read from Overture's public S3 bucket). Foursquare Open Source Places, release `2026-09-15` (`FSQ_RELEASE`, read from Hugging Face). The Foursquare dataset is gated: accept its terms on [Hugging Face](https://huggingface.co/datasets/foursquare/fsq-os-places) and set `HF_TOKEN` in your environment before running step 1. The first cells of step 1 scan about 11 GB of Overture data and take several minutes. Each one skips itself once its output exists.

**Training time** (Apple M1 Pro laptop; epochs run = all epochs actually run, from the saved histories):

| Model | Device | Per epoch | Runs | Epochs run |
|---|---|---|---|---|
| Classical (3 models × 2 settings) | CPU | (none) | deterministic | about 1 min + feature build |
| Siamese BiLSTM | CPU | about 4 min | 2 poolings × 3 seeds, ≤ 20 epochs | 71 |
| ESIM | CPU | about 3 min | 5 settings × 3 seeds, ≤ 20 epochs | 133 |
| DistilBERT bi-encoder | GPU (MPS) | about 17 min | 3 seeds, ≤ 5 epochs | 15 |
| DistilBERT cross-encoder | GPU (MPS) | about 14 min | 3 seeds, ≤ 5 epochs | 15 |

Retraining gives close but not identical numbers: GPU/MPS kernels are not bit-for-bit deterministic, and gensim's word vectors depend on `PYTHONHASHSEED`, which the paper's runs did not fix.

## Paper → notebook section

All in `results/paper_tables.ipynb`:

| Paper | Section |
|---|---|
| Table 1: pair types, test counts; Figure 1: data pipeline (`tools/figure_pipeline.py`); Table 2: the 23 features | §1 |
| Tables 3, 4: classical models, sentinel vs zero-fill; Tables A1, A2: random-forest importance | §2 |
| Tables 5, 6, 7: siamese BiLSTM | §3 |
| Table 8: every model on the rare-number group; Table 9: word2vec `min_count` 1 per seed; Table 10: ESIM settings (val); Table 11: ESIM vs siamese | §4 |
| Table 12: bi- vs cross-encoder; Table 13: AUC by house-number group; Figure 3 and Table 14: Youden's J; Table 15: every kept model | §5 |
| Table B1: missing-number matches found, earlier version vs this one | §6 |
| Numbers quoted in the Discussion | §7 |

The cosine check quoted in the ESIM section (same-street rare house numbers vs random number pairs) is `esim/01_word2vec.ipynb` §4. It needs the word-vector tables.

## Layout

```
helpers/      pairs (load_pairs, add_number_street), similarity features, slice tables, paths, look-alike street checks,
              released.py (load the Hugging Face weights)
pipeline/     data notebooks 01–04; Data/ (git-ignored)
classical/    LogReg / random forest / XGBoost; features/ cache (git-ignored)
siamese/      model.py, training.py (the training protocol shared by every neural model), train.ipynb, train_seeds.py
esim/         model.py, training.py, 01_word2vec.ipynb, 02_train.ipynb
distilbert/   model.py, training.py, train_bi.ipynb, train_cross.ipynb, train.py
predictions/  per run: probs_{val,test}_seed_{s}.npy + history_seed_{s}.csv; rare_number_group_{val,test}.npy
checkpoints/  trained weights and word-vector tables (git-ignored)
results/      paper_tables.ipynb, v1_missing_number_matches.csv (Table B1), figures/
tools/        figure_pipeline.py (Figure 1), classical_seeds.py, run_notebooks.py
paper/        the paper (PDF)
```

## Data licences and attribution

This repository contains no source data: the pair files are rebuilt from the public releases by `pipeline/`, or downloaded from the Hugging Face dataset (which ships Foursquare's `NOTICE.txt` and the Apache 2.0 licence). The pairs carry the source licences:

- **Overture Maps Places** (release `2026-09-23.1`) is published under [CDLA Permissive 2.0](https://cdla.dev/permissive-2-0/) and Apache 2.0, depending on the source of each record ([Overture attribution](https://docs.overturemaps.org/attribution)). Side A of every pair is the address Overture publishes, mostly from Meta and Microsoft: *Data from Meta. Available under CDLA Permissive 2.0.* *Data from Microsoft. Available under CDLA Permissive 2.0.* Overture's suggested citation for publications: *Overture Maps Foundation, overturemaps.org* (accessed 2026-09).
- **Overture bridge files** (GERS ID → source record IDs, same release): Overture's documentation states no separate licence for them.
- **Foursquare Open Source Places** (release `2026-09-15`): *Copyright 2024 Foursquare Labs, Inc. All rights reserved.* Available under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) ([NOTICE](https://opensource.foursquare.com/places-notice-txt/)). Side B of every pair is the Foursquare address, changed by `pipeline/01_overture_fsq.ipynb` (units, floors and punctuation removed, ZIP codes normalised) and, for look-alike street pairs, by `pipeline/03_case2s.ipynb` (street replaced by another street's text).

## Code licence

MIT (see `LICENSE`).

## Citation

```bibtex
@misc{zhang2026addressmatching,
  author = {Zhang, David},
  title  = {Address Matching by Pair Type: Segmented Classical Models versus Raw-String Neural Matchers},
  year   = {2026},
  howpublished = {\url{https://github.com/DZhang3210/address-matching-by-pair-type}}
}
```
