# Address matching by pair type: classical models vs. neural matchers

Code and saved model outputs for the paper *(title, link)*. Given two address strings, each model predicts whether they refer to the same location. We compare classical models on hand-segmented addresses (logistic regression, random forest and XGBoost on 23 string-similarity features) with four neural models that read the raw strings (a character-level siamese BiLSTM, ESIM, and DistilBERT bi- and cross-encoders). All of them are trained and tested on the same splits of real cross-source address pairs from Connecticut, built from Overture Maps Places and Foursquare Open Source Places, and every model is scored separately on each pair type. The repository holds only what the paper uses: the data pipeline, the model code, and one notebook that reproduces every table.

## Setup

Python 3.9 (the paper's runs used 3.9.6).

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`postal` (the Python binding for libpostal) needs the libpostal C library installed first; see the [libpostal README](https://github.com/openvenues/libpostal#installation-maclinux). Only the data pipeline needs it (`pipeline/03_case2s.ipynb` and `04_case2s_checks.ipynb` use it to flag two spellings of the same street).

Run every notebook from its own folder (each one adds the repository root to `sys.path`).

## Reproducing the tables (no retraining)

The val/test probabilities of every neural run are committed in `predictions/`, so the tables need only the pair files:

1. Rebuild the pairs: `pipeline/01` → `04` (below).
2. Run `results/paper_tables.ipynb`. It refits the three classical models from the similarity features (deterministic; the first run builds the feature cache in a few minutes) and reads every neural run from `predictions/`.

## Run order (full rebuild)

| Step | Notebook | Writes (git-ignored unless noted) |
|---|---|---|
| 1 | `pipeline/01_overture_fsq.ipynb`: Overture places in CT, bridge files, Foursquare records, join, address and ZIP cleaning (DuckDB) | `pipeline/Data/ct_positives_*` |
| 2 | `pipeline/02_build_pairs.ipynb`: ZIP-based split, case 1, case 2 and case 3 negatives, 1:9 ratio | `pipeline/Data/ct_pairs_{split}_2026-09-23.1.parquet` |
| 3 | `pipeline/03_case2s.ipynb`: look-alike street swaps (case 2s) replace some easy negatives (**the paper's dataset**) | `pipeline/Data/ct_pairs_{split}_case2s.parquet` |
| 4 | `pipeline/04_case2s_checks.ipynb`: acceptance checks on the case 2s files | none |
| 5 | `siamese/train.ipynb`: siamese BiLSTM, mean and max pooling, seed 0 | `checkpoints/siamese/`, `predictions/siamese_*` (committed) |
| 6 | `esim/01_word2vec.ipynb`: word2vec and fastText tables | `checkpoints/esim/*.model` |
| 7 | `esim/02_train.ipynb`: ESIM, five embedding settings × 3 seeds, plus the rare-number group masks | `checkpoints/esim/`, `predictions/esim_*` and `rare_number_group_*.npy` (committed) |
| 8 | `distilbert/train_bi.ipynb`, `distilbert/train_cross.ipynb`: 3 seeds each | `checkpoints/distilbert_*`, `predictions/distilbert_*` (committed) |
| 9 | `results/paper_tables.ipynb`: every table | none |

**Data releases.** Overture Maps Places and bridge files, release `2026-09-23.1` (`OVERTURE_RELEASE`, read from Overture's public S3 bucket). Foursquare Open Source Places, release `2026-09-15` (`FSQ_RELEASE`, read from Hugging Face). The Foursquare dataset is gated: accept its terms on [Hugging Face](https://huggingface.co/datasets/foursquare/fsq-os-places) and set `HF_TOKEN` in your environment before running step 1. The first cells of step 1 scan about 11 GB of Overture data and take several minutes. Each one skips itself once its output exists.

**Training time** (from the paper's Experimental Setup; total = all epochs actually run, from the saved histories):

| Model | Hardware | Per epoch | Runs | Total |
|---|---|---|---|---|
| Classical (3 models × 2 settings) | M1 Pro laptop | (none) | deterministic | about 1 min + feature build |
| Siamese BiLSTM | M1 Pro (CPU) | about 4 min | 2 poolings × 1 seed, ≤ 20 epochs | about 1.3 h |
| ESIM | M1 Pro (CPU) | about 3 min | 5 settings × 3 seeds, ≤ 20 epochs | about 6.5 h |
| DistilBERT bi-encoder | Google Colab T4 | about 10 min | 3 seeds, ≤ 5 epochs | about 2.3 h |
| DistilBERT cross-encoder | M1 Pro (MPS) | about 16 min | 3 seeds, ≤ 5 epochs | about 3.5 h |

Retraining gives close but not identical numbers: GPU/MPS kernels are not bit-for-bit deterministic, and gensim's word vectors depend on `PYTHONHASHSEED`, which the paper's runs did not fix.

## Paper table → notebook section

All in `results/paper_tables.ipynb`:

| Paper table | Section |
|---|---|
| Table 1: pair types, test counts | §1 |
| Table 2: the 23 features | `helpers/features.py` (definition, no output) |
| Table 3: classical models, sentinel vs zero-fill | §2, "Table 3" |
| Table 4: one side missing a house number | §2, "Table 4" |
| Table 5: siamese BiLSTM by pooling | §3, "Table 5" |
| Table 6: siamese, one side missing | §3, "Table 6" |
| Table 7: siamese vs tree models | §3, "Table 7" |
| Table 8: ESIM word2vec `min_count` 1 per seed, rare-number group | §4, "Table 8" |
| Table 9: ESIM settings on the rare-number group (val) | §4, "Table 9" |
| Table 10: ESIM settings by pair type (val) | §4, "Table 10" |
| Table 11: ESIM vs siamese (test) | §4, "Table 11" |
| Table 12: DistilBERT bi- vs cross-encoder | §5, "Table 12" |
| Table 13: AUC by house-number group | §5, "Table 13" |
| Table 14: every model on one AUC | §5, "Table 14" |
| Tables A1, A2: random-forest importance | §2, "Tables A1, A2" |
| Discussion: matches vs case 1, one side missing | §6 |

The cosine check quoted in the ESIM section (same-street rare house numbers vs random number pairs) is `esim/01_word2vec.ipynb` §4. It needs the word-vector tables.

## Layout

```
helpers/      pairs (load_pairs, add_number_street), similarity features, slice tables, paths, case 2s checks
pipeline/     data notebooks 01–04; Data/ (git-ignored)
classical/    LogReg / random forest / XGBoost; features/ cache (git-ignored)
siamese/      model.py, training.py (the training protocol shared by every neural model), train.ipynb
esim/         model.py, training.py, 01_word2vec.ipynb, 02_train.ipynb
distilbert/   model.py, training.py, train_bi.ipynb, train_cross.ipynb
predictions/  per run: probs_{val,test}_seed_{s}.npy + history_seed_{s}.csv; rare_number_group_{val,test}.npy
checkpoints/  trained weights and word-vector tables (git-ignored)
results/      paper_tables.ipynb
```

## Data licences and attribution

This repository contains no source data: the pair files are rebuilt from the public releases by `pipeline/`. Pairs built with it carry the source licences:

- **Overture Maps Places** (release `2026-09-23.1`) is published under [CDLA Permissive 2.0](https://cdla.dev/permissive-2-0/) and Apache 2.0, depending on the source of each record ([Overture attribution](https://docs.overturemaps.org/attribution)). Side A of every pair is the address Overture publishes, mostly from Meta and Microsoft: *Data from Meta. Available under CDLA Permissive 2.0.* *Data from Microsoft. Available under CDLA Permissive 2.0.* Overture's suggested citation for publications: *Overture Maps Foundation, overturemaps.org* (accessed 2026-09).
- **Overture bridge files** (GERS ID → source record IDs, same release): Overture's documentation states no separate licence for them.
- **Foursquare Open Source Places** (release `2026-09-15`): *Copyright 2024 Foursquare Labs, Inc. All rights reserved.* Available under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) ([NOTICE](https://opensource.foursquare.com/places-notice-txt/)). Side B of every pair is the Foursquare address, changed by `pipeline/01_overture_fsq.ipynb` (units, floors and punctuation removed, ZIP codes normalised) and, for case 2s pairs, by `pipeline/03_case2s.ipynb` (street replaced by another street's text).

## Code licence

MIT (see `LICENSE`).

## Citation

*(paper reference)*
