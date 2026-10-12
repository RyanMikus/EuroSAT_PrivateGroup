"""Shared building blocks of the experiment notebooks (notebooks/experiments/).

Each experiment notebook only sets its configuration and calls these functions, so the one
thing an experiment changes is visible at the top of the notebook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from src.data import CLASSES, PROJECT_ROOT, band_index, load_split, load_test, load_train
from src.plot_style import SPLIT_COLORS, TEST_COLOR, TRAIN_COLOR
from src.preprocessing import land_mask, load_matcher
from src.training import adapt_batchnorm, predict_proba, to_device_images

PREPROCESSING_DIR = PROJECT_ROOT / "artifacts" / "preprocessing"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_correction(name: str):
    """Load a saved correction from artifacts/preprocessing/<name>.json.

    Histogram matchers come from 04_haze_correction, spectral matching from 05_fourier_analysis.
    """
    path = PREPROCESSING_DIR / f"{name}.json"
    if "land_ratio" in json.loads(path.read_text()):
        from src.fourier import SpectralMatcher
        return SpectralMatcher.load(path)
    return load_matcher(path)


@dataclass
class ExperimentData:
    """All images of one experiment: NumPy arrays (uint16) and the same images on the device."""

    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    X_kg: np.ndarray
    test_ids: pd.DataFrame
    tr_images: torch.Tensor
    tr_labels: torch.Tensor
    va_images: torch.Tensor
    va_labels: torch.Tensor
    te_images: torch.Tensor
    kg_images: torch.Tensor

    def summary(self) -> str:
        return (f"train {len(self.X_tr):,}  val {len(self.X_va):,}  internal test {len(self.X_te):,}  "
                f"Kaggle test {len(self.X_kg):,}")


def load_experiment_data(bands: list[str], corrections: list[str], device: torch.device) -> ExperimentData:
    """Load the split, apply the corrections (in order, on all 12 bands) to every EuroSAT image,
    select ``bands`` and move everything to ``device``. Kaggle images are never corrected."""
    X_all, _ = load_train()
    X_kaggle, test_ids = load_test()
    split = load_split()
    band_idx = [band_index(b) for b in bands]
    steps = [load_correction(name) for name in corrections]

    def subset(name):
        rows = np.flatnonzero(split["split"].values == name)
        x = np.asarray(X_all[rows])
        for step in steps:
            x = step(x)
        return x[:, band_idx], split["label_id"].values[rows]

    X_tr, y_tr = subset("train")
    X_va, y_va = subset("val")
    X_te, y_te = subset("test")
    X_kg = np.asarray(X_kaggle[:, band_idx])
    return ExperimentData(
        X_tr, y_tr, X_va, y_va, X_te, y_te, X_kg, test_ids,
        tr_images=to_device_images(X_tr, device), tr_labels=torch.from_numpy(y_tr).to(device),
        va_images=to_device_images(X_va, device), va_labels=torch.from_numpy(y_va).to(device),
        te_images=to_device_images(X_te, device), kg_images=to_device_images(X_kg, device),
    )


# ----------------------------------------------------------------------------
# Monitoring and measurements without labels
# ----------------------------------------------------------------------------

def kaggle_monitor(data: ExperimentData, normalizer):
    """Per-epoch measurements on the unlabelled Kaggle images (see src.training.fit)."""
    def monitor(model):
        val_p = predict_proba(model, data.va_images, normalizer)
        kg_p = predict_proba(model, data.kg_images, normalizer)
        return {
            "val_conf": float(val_p.max(1).mean()),
            "kg_conf": float(kg_p.max(1).mean()),
            "kg_sealake": float((kg_p.argmax(1) == CLASSES.index("SeaLake")).mean()),
            **{f"kg_share_{c}": float((kg_p.argmax(1) == i).mean()) for i, c in enumerate(CLASSES)},
        }
    return monitor


def class_mix(prob: np.ndarray) -> pd.Series:
    """Share of predicted classes in %."""
    return pd.Series(np.bincount(prob.argmax(1), minlength=10) / len(prob) * 100, index=CLASSES)


def view_agreement(views: np.ndarray) -> float:
    """Share of images whose 8 rotated/flipped views all get the same class; views: (8, N, 10)."""
    per_view = views.argmax(2)
    return float((per_view == per_view[0]).all(0).mean())


# ----------------------------------------------------------------------------
# AdaBN with the training water share
# ----------------------------------------------------------------------------

def balanced_kaggle_subset(X_tr: np.ndarray, X_kg: np.ndarray, seed: int) -> np.ndarray:
    """Indices of all Kaggle land images plus enough water images to match the training water share."""
    water_share = 1 - land_mask(X_tr).mean()
    kg_land = land_mask(X_kg)
    n_water = int(round(kg_land.sum() * water_share / (1 - water_share)))
    rng = np.random.default_rng(seed)
    return np.sort(np.r_[np.flatnonzero(kg_land), rng.choice(np.flatnonzero(~kg_land), n_water, replace=False)])


def adabn_balanced(model, normalizer, data: ExperimentData, seed: int):
    """Re-measure the BatchNorm statistics on the balanced Kaggle subset (changes ``model``)."""
    idx = balanced_kaggle_subset(data.X_tr, data.X_kg, seed)
    adapt_batchnorm(model, data.kg_images[torch.from_numpy(idx).to(data.kg_images.device)], normalizer)
    return model, idx


def find_run(name_suffix: str) -> Path:
    """Most recent run folder whose name ends with ``_<name_suffix>``."""
    matches = sorted((PROJECT_ROOT / "runs").glob(f"*_{name_suffix}"))
    if not matches:
        raise FileNotFoundError(f"no run folder ending in _{name_suffix}")
    return matches[-1]


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def plot_training_curves(history: pd.DataFrame) -> None:
    """Loss, accuracy and mean confidence (validation vs Kaggle) per epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    e = history["epoch"]
    axes[0].plot(e, history["train_loss"], color=TRAIN_COLOR, label="Train")
    axes[0].plot(e, history["val_loss"], color=SPLIT_COLORS["val"], label="Validation")
    axes[0].set_title("Loss", loc="left")
    axes[1].plot(e, history["train_acc"] * 100, color=TRAIN_COLOR, label="Train")
    axes[1].plot(e, history["val_acc"] * 100, color=SPLIT_COLORS["val"], label="Validation")
    axes[1].set_title("Accuracy (%)", loc="left")
    axes[2].plot(e, history["val_conf"] * 100, color=SPLIT_COLORS["val"], label="Validation")
    axes[2].plot(e, history["kg_conf"] * 100, color=TEST_COLOR, ls="--", label="Kaggle test (no labels)")
    axes[2].set_title("Mean confidence (%)", loc="left")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    share = cm / cm.sum(axis=1, keepdims=True)
    blues = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(8, 6.8))
    ax.imshow(share, cmap=blues, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if share[i, j] > 0.5 else "#52514e")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left")
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_class_mix(mix: pd.DataFrame, title: str = "Predicted class mix on the Kaggle test set") -> None:
    """Grouped bars, one column of ``mix`` per series (values in %). Up to 3 series."""
    colors = [TRAIN_COLOR, TEST_COLOR, "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = np.arange(len(mix))
    w = 0.8 / len(mix.columns)
    for i, col in enumerate(mix.columns):
        ax.bar(x + (i - (len(mix.columns) - 1) / 2) * w, mix[col], w - 0.03, color=colors[i], label=col)
    ax.set_xticks(x, mix.index, rotation=20, ha="right")
    ax.set_ylabel("Share of images (%)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(mix.columns))
    plt.tight_layout()
    plt.show()
