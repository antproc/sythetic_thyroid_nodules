
"""Compute mean and std of the grayscale training set."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from config import TRAIN_DIR

def compute_dataset_stats(data_dir: Path):
    paths = [p for p in data_dir.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg"}]
    to_tensor = ToTensor()
    total = total_sq = n_pix = 0.0
    for p in paths:
        img = Image.open(p).convert("L")
        t = to_tensor(img)
        total += t.sum().item()
        total_sq += (t**2).sum().item()
        n_pix += t.numel()
    mean = total / n_pix
    var = total_sq / n_pix - mean**2
    return mean, np.sqrt(var)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(TRAIN_DIR))
    args = parser.parse_args()
    m, s = compute_dataset_stats(Path(args.data_dir))
    print(f"Dataset mean: {m:.4f}\nDataset std : {s:.4f}")
