from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DATA_DIR = PROJECT_ROOT / "data" / "kaggle"
EUROSAT_MS_DIR = PROJECT_ROOT / "data" / "eurosat" / "ms"
EUROSAT_RGB_DIR = PROJECT_ROOT / "data" / "eurosat" / "rgb"
KAGGLE_FILES = [
    "train.csv",
    "test.csv",
    "sample_submission.csv",
]


def inspect_csv(file_path: Path) -> None:
    """Print a compact overview of one Kaggle CSV file."""
    print(f"\nFile: {file_path.name}")

    if not file_path.exists():
        print(f"Missing: expected file at {file_path}")
        return

    data = pd.read_csv(file_path)

    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("First rows:")
    print(data.head())

    if file_path.name == "train.csv" and "label" in data.columns:
        print("Label distribution:")
        print(data["label"].value_counts())


def main() -> None:
    """Inspect expected project data locations and Kaggle CSV files."""
    print("Dataset directories:")
    for label, path in [
        ("EuroSAT multispectral", EUROSAT_MS_DIR),
        ("EuroSAT RGB", EUROSAT_RGB_DIR),
        ("Kaggle competition data", KAGGLE_DATA_DIR),
    ]:
        status = "exists" if path.exists() else "missing"
        print(f"- {label}: {path} ({status})")

    for file_name in KAGGLE_FILES:
        inspect_csv(KAGGLE_DATA_DIR / file_name)


if __name__ == "__main__":
    main()
