
"""Generate synthetic nodules via in-painting."""
from pathlib import Path
from typing import Tuple
import numpy as np, torch, cv2
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from config import CACHE_DIR, GUIDANCE_SCALE, INPAINT_DIR, MODEL_ID, TRAIN_DIR
from utils import is_empty_image

def _transition_masks(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inner, outer = 5, 15
    inner_m = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner, inner)),1)
    outer_m = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer, outer)),1)
    m_f = mask.astype(float)/255.0
    trans = np.where(m_f>0.5, gaussian_filter(m_f,2.0), gaussian_filter(m_f,3.0))
    return inner_m, outer_m, np.clip(trans,0,1)

def _blend(orig: np.ndarray, synth: np.ndarray, trans: np.ndarray)->np.ndarray:
    return (orig*(1-trans[...,None]) + synth*trans[...,None]).astype(np.uint8)

def generate(start_index:int=0):
    device="cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, low_cpu_mem_usage=True).to(device)
    ft_unet = UNet2DConditionModel.from_pretrained("thyroid_node_unet").to(device)
    pipe.unet = ft_unet
    imgs = [p for p in Path(TRAIN_DIR).glob("*.jpg") if not p.stem.endswith("_mask")][start_index:]
    out_dir = Path(INPAINT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    for p in tqdm(imgs, desc="in-paint"):
        mask_p = p.with_name(p.stem + "_mask.jpg")
        if not mask_p.exists(): continue
        img = Image.open(p).convert("RGB")
        mask = Image.open(mask_p).convert("L")
        m_np = (np.asarray(mask)>128).astype(np.uint8)*255
        _, outer, trans = _transition_masks(m_np)
        for _ in range(15):
            out = pipe(prompt="Thyroid nodule, ultrasound imaging, clear detail",
                       image=img, mask_image=Image.fromarray(outer),
                       num_inference_steps=100, guidance_scale=GUIDANCE_SCALE).images[0]
            if not is_empty_image(out): break
        res = _blend(np.asarray(img), np.asarray(out), trans)
        Image.fromarray(res).save(out_dir/f"{p.stem}_inpainted.jpg")
        mask.save(out_dir/f"{p.stem}_inpainted_mask.jpg")
if __name__=="__main__":
    generate()
