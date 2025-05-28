
"""Central configuration – adjust paths and hyper-parameters here."""

from pathlib import Path


TRAIN_DIR = "data_train_resize_512"

GUIDANCE_SCALE = 20
SYNTH_DIR = Path(f"synthetic_nodes_512_custom_norm_guidance_{GUIDANCE_SCALE}")
INPAINT_DIR = SYNTH_DIR

MODEL_ID = "CompVis/stable-diffusion-v1-4"
CACHE_DIR = Path("models_cache")

MEAN, STD = 0.2026, 0.2910

BATCH_SIZE = 1
LR = 5e-6
EPOCHS = 4

INFERENCE_STEPS = 100
