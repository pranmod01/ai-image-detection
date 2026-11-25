"""Shared dataset: reads pipeline metadata CSVs and turns rows into (image, target).

Images are already standardized on disk, so the only transform is to a float
tensor in [0, 1]. Each detector does its own input normalization (ImageNet
stats, DCT, CLIP preprocessing), so data loading is identical for every model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.contract import CLASS_TO_IDX, CLASSES, METADATA_COLUMNS


def load_metadata(paths: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p, dtype={"strength": float, "mask_area": float})
                    for p in paths], ignore_index=True)
    missing = set(METADATA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"metadata is missing contract columns: {sorted(missing)}")
    unknown = set(df["class"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown classes in metadata: {sorted(unknown)}")
    if df["image_id"].duplicated().any():
        raise ValueError("duplicate image_id across metadata files")
    return df


def add_targets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Adds a `target` column for the configured label setup; drops rows without one."""
    df = df.copy()
    if cfg["label_setup"] == "ordinal":
        df["target"] = df["class"].map(CLASS_TO_IDX).astype(float)
        return df
    if cfg["label_setup"] == "regression":
        r = cfg["regression"]
        cont = df["class"].isin(r["continuous_classes"])
        df["target"] = np.where(cont, df["strength"].fillna(df["mask_area"]), np.nan)
        fixed = df["class"].map(r["fixed_targets"])
        df["target"] = df["target"].fillna(fixed)
        return df.dropna(subset=["target"]).reset_index(drop=True)
    raise ValueError(f"unknown label_setup {cfg['label_setup']!r}")


class EditDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str):
        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = np.asarray(Image.open(self.root / r["path"]).convert("RGB"))
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255
        return x, torch.tensor(r["target"], dtype=torch.float32), i
