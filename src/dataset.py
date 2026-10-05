"""Dataset scaffolding for EuroSAT image loading.

This module will hold the PyTorch dataset implementation once the exact local
file layout and image formats are confirmed.
"""


class EuroSATDataset:
    """Placeholder for a future PyTorch EuroSAT dataset loader."""

    def __init__(self) -> None:
        # TODO: Load EuroSAT training images from data/eurosat/.
        # TODO: Support labels from train.csv and/or directory structure.
        # TODO: Handle 64x64 Sentinel-2 images.
        # TODO: Consider support for all 13 Sentinel-2 spectral bands.
        # TODO: Add transforms and preprocessing hooks.
        # TODO: Confirm image file formats before implementing loading logic.
        raise NotImplementedError("EuroSATDataset will be implemented later.")
