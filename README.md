# EuroSAT Land-Cover Classification

Land-cover classification of 64 × 64 Sentinel-2 image patches into 10 classes, for the
Machine Learning coding challenge at the University of St. Gallen (Kaggle competition
`7-854-1-00-machine-learning-2026-coding-challenge`).

**The core difficulty:** the training data (EuroSAT) is Sentinel-2 **Level-1C**, with haze,
while the Kaggle test data is **Level-2A**, atmospherically corrected. Models reach about 99% on
EuroSAT but far less on Kaggle. Most of the work in this repository is about closing that gap.

## Results so far

| Run | Change | Internal test | Kaggle public |
|---|---|---|---|
| `2026-10-12_0824_baseline_effb0` | EfficientNet-B0, 12 bands, no correction | 98.79% | 56.48% |
| `2026-10-12_0910_hist_split_effb0` | + histogram matching of training images (`hist_split`) | 98.64% | **64.06%** |

Both models have 4,022,950 parameters (medium track, limit 10M). Details per run are in `runs/`.

## Repository structure

```
notebooks/     the project, step by step (run in order, see below)
src/           shared Python code imported by the notebooks
scripts/       data download scripts
artifacts/     small results of the data notebooks, shared via Git
  splits/        train/val/test split (03_split)
  preprocessing/ fitted haze corrections and their comparison (05_preprocessing)
  quality_flags.csv  data quality flags (02_data_quality)
runs/          one folder per training run, shared via Git (weights only for submitted runs)
data/          raw data and caches, NOT in Git (download with scripts/)
```

**Rule of thumb:** `data/` holds what can be downloaded again. `artifacts/` and `runs/` hold what
we created ourselves and need to share.

### Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Exploratory analysis: class balance, band order, brightness and resolution differences between train and test |
| `02_data_quality.ipynb` | Quality checks (duplicates, broken bands, outliers, haze). Nothing removed, unusual images flagged |
| `03_split.ipynb` | Stratified 70/15/15 train/val/test split; neighbouring patches kept in the same split |
| `04_baseline.ipynb` | Baseline model, reference for all improvements |
| `05_preprocessing.ipynb` | Compares haze corrections (linear, histogram matching variants). No training needed |
| `06_train_hist_split.ipynb` | Baseline with the best correction (`hist_split`) |

Training notebooks have a `LOAD_RUN` setting: with a run folder set, they skip training, load the
saved weights (`model.pt` from Git) and recompute all results and the exact submission file in
about a minute. Set `LOAD_RUN = None` to train a new model; retraining gives a very similar but not
identical model (GPU calculations are not bit-for-bit reproducible), so Kaggle scores can differ
by about 1–2 points.

### `src/`

| File | Content |
|---|---|
| `data.py` | Band orders and mapping, class list, paths, loading (with a fast `.npy` cache in `data/cache/`) |
| `preprocessing.py` | Haze corrections: histogram matching (`HistogramMatcher`, `SplitMatcher`), linear percentile matching |
| `models.py` | Model builders; adapts pretrained RGB weights to 12 bands; track parameter limits |
| `training.py` | GPU training loop, early stopping, prediction with test-time augmentation, saving and loading runs |
| `utils.py` | Device selection (CUDA → Apple GPU → CPU), seeding, parameter count |
| `plot_style.py` | Shared plot colours and style |

### A run folder

```
runs/<date>_<name>/
  config.json     model, bands, preprocessing, normalisation, hyperparameters
  history.csv     loss, accuracy and Kaggle monitoring per epoch
  metrics.json    validation / internal test accuracy, Kaggle public score
  submission.csv  the file submitted to Kaggle
  model.pt        trained weights (16 MB; in Git only for runs submitted to Kaggle)
```

## Setup

macOS, from the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name eurosat --display-name "EuroSAT (.venv 3.14)"
```

In VS Code, select the kernel **EuroSAT (.venv 3.14)** for the notebooks.

### Kaggle token

1. Join the competition on Kaggle and accept its rules.
2. Kaggle → Settings → API → **Generate New Token**. Copy the token value (starts with `KGAT_`;
   it is shown only once). The token *name* is not the token.
3. Save it, without echoing it to the terminal:

```bash
mkdir -p ~/.kaggle
read -s "TOKEN?Paste Kaggle token: " && printf '%s' "$TOKEN" > ~/.kaggle/access_token && unset TOKEN
chmod 600 ~/.kaggle/access_token
```

Never commit tokens. (`kagglehub.login()` only keeps the token for the current Python process,
so it does not work for the download scripts.)

### Download data

```bash
python scripts/download_data.py --ms-only      # EuroSAT 13-band GeoTIFFs (~2 GB)
python scripts/download_data.py --kaggle-only  # Kaggle test set and CSVs
```

| Command | Description |
| --- | --- |
| `python scripts/download_data.py` | Everything (also the RGB version, which the notebooks don't need) |
| `python scripts/download_data.py --eurosat-only` | Both EuroSAT datasets |
| `python scripts/download_data.py --rgb-only` | Only the RGB EuroSAT dataset |
| `python scripts/download_data.py --force` | Force re-download |

The first notebook run builds `data/cache/` (about 3 GB, one minute).

## Hardware

Training uses the GPU automatically: CUDA if available, otherwise the Apple GPU (MPS), otherwise
the CPU. On an M4 MacBook one training run (20 epochs) takes about 18 minutes; on the CPU it would
take days. The MacBook Air throttles under long load, so expect roughly 20–40% longer.

Distributing one run over several laptops is not worth it. Run **different experiments in
parallel**, one per teammate.

## Submitting to Kaggle

```bash
kaggle competitions submit -c 7-854-1-00-machine-learning-2026-coding-challenge \
  -f runs/<run>/submission.csv -m "<what changed>"
kaggle competitions submissions -c 7-854-1-00-machine-learning-2026-coding-challenge
```

Limit: 15 submissions per day. Write the public score into the run's `metrics.json`.

## Team workflow

- **One change per experiment**, each in its own numbered notebook and run folder, so every
  Kaggle score can be traced to exactly one change.
- **Don't edit the same notebook at the same time**: notebooks are hard to merge in Git.
- Notebooks are committed **with outputs** (they are the documentation). Run them top to bottom
  before committing.
- **Weights of every run submitted to Kaggle go into Git**: add a `!runs/<run>/model.pt` line to
  `.gitignore`. Then anyone can reproduce the exact model and score with `LOAD_RUN`. Weights of
  other runs stay local.

## Data sources

- EuroSAT, official Zenodo record: https://zenodo.org/records/7711810
  (`EuroSAT_MS.zip`: 13 Sentinel-2 bands, 27,000 labelled images, 10 classes). MIT license,
  based on Copernicus Sentinel data.
- Kaggle competition `7-854-1-00-machine-learning-2026-coding-challenge`: Level-2A test set
  (4,232 images, 12 bands without B10) and submission format.

## References

1. Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). "EuroSAT: A Novel Dataset and Deep
   Learning Benchmark for Land Use and Land Cover Classification." IEEE JSTARS.
2. Helber, P., Bischke, B., Dengel, A., & Borth, D. (2018). "Introducing EuroSAT: A Novel Dataset
   and Deep Learning Benchmark for Land Use and Land Cover Classification." IGARSS 2018.
3. Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural
   Networks." ICML. ImageNet weights from torchvision.
4. Last year's solution (team M.A.R.S, 2025) was used as a reference for the problem analysis.
   All code in this repository was written independently.
