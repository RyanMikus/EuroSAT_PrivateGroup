# EuroSAT Land-Cover Classification

This repository contains code for land-cover classification using EuroSAT and
Sentinel-2 imagery. The task is single-label classification of 64x64 satellite
image patches into 10 land-cover classes.

EuroSAT provides the labelled training data. Kaggle provides the competition
test data and submission format. Both the 13-band multispectral EuroSAT dataset
and the RGB version can be downloaded using the included scripts.

This project is used for a Machine Learning coding challenge at the University
of St. Gallen.

## Setup

Assume macOS and that you are in the repository root.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Kaggle Authentication

Kaggle authentication is required for the competition data.

1. Log into Kaggle.
2. Open Kaggle Settings.
3. Generate an API token.
4. Authenticate locally before downloading the Kaggle competition files.

This repository uses KaggleHub:

```bash
python -c "import kagglehub; kagglehub.login()"
```

KaggleHub will prompt for the API token. Never commit Kaggle API tokens or other
credentials to Git. You may also need to join the competition and accept its
rules on Kaggle before the competition files can be downloaded.

## Download Data

Download all required data:

```bash
python scripts/download_data.py
```

This downloads EuroSAT multispectral data, EuroSAT RGB data, and Kaggle
competition data.

| Command | Description |
| --- | --- |
| `python scripts/download_data.py --eurosat-only` | Download both EuroSAT datasets |
| `python scripts/download_data.py --ms-only` | Download only the 13-band multispectral EuroSAT dataset |
| `python scripts/download_data.py --rgb-only` | Download only the RGB EuroSAT dataset |
| `python scripts/download_data.py --kaggle-only` | Download only Kaggle competition data |
| `python scripts/download_data.py --force` | Force re-download of existing datasets |

The individual download scripts can also be run directly:

```bash
python scripts/download_eurosat.py
python scripts/download_kaggle.py
```

## Inspect the Data

```bash
python src/inspect_data.py
```

This prints useful information about the downloaded datasets and Kaggle CSV
files, including shapes, columns, sample rows, and labels where available.

## Data Sources

EuroSAT official Zenodo record:
https://zenodo.org/records/7711810

- `EuroSAT_MS.zip` - multispectral version with all 13 Sentinel-2 bands
- `EuroSAT_RGB.zip` - RGB representation
- 27,000 labelled images
- 10 land-cover classes

Kaggle competition handle:
`7-854-1-00-machine-learning-2026-coding-challenge`

Kaggle provides the held-out competition test set and submission files.

## References

1. Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).
   "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land
   Cover Classification." IEEE Journal of Selected Topics in Applied Earth
   Observations and Remote Sensing.
2. Helber, P., Bischke, B., Dengel, A., & Borth, D. (2018).
   "Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land
   Use and Land Cover Classification." IGARSS 2018.

## Data License

EuroSAT is distributed under the MIT license and is based on publicly available
Copernicus Sentinel data. See the official Zenodo record for further details:
https://zenodo.org/records/7711810
