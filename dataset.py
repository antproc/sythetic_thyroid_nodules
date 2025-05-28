
"""PyTorch Dataset for thyroid nodule ultrasound images."""

from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import MEAN, STD, TRAIN_DIR

class ThyroidNodeDataset(Dataset):
    def __init__(self, root: Path = TRAIN_DIR):
        self.image_paths = [p for p in Path(root).glob("*.jpg") if not p.stem.endswith("_mask")]
        self.img_tf = transforms.Compose([
            transforms.Resize((512,512), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([MEAN],[STD])
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((512,512), transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx:int)->Dict[str,Any]:
        img_path = self.image_paths[idx]
        mask_path = img_path.with_name(img_path.stem + "_mask.jpg")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST)
        m_arr = np.array(mask) > 128
        m_img = np.array(img)
        m_img[~m_arr] = 0
        return {
            "image": self.img_tf(img),
            "masked_image": self.img_tf(Image.fromarray(m_img)),
            "mask": self.mask_tf(mask),
            "image_path": str(img_path)
        }
