
"""Fine-tune Stable-Diffusion U-Net on thyroid nodules."""
from pathlib import Path
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from config import BATCH_SIZE, CACHE_DIR, EPOCHS, LR, MODEL_ID, TRAIN_DIR
from dataset import ThyroidNodeDataset

def finetune_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", cache_dir=CACHE_DIR)
    text_enc = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", cache_dir=CACHE_DIR).to(device)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", cache_dir=CACHE_DIR).to(device)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, low_cpu_mem_usage=True)
    unet = pipe.unet.to(device)
    vae.eval(); text_enc.eval()
    for p in (*vae.parameters(), *text_enc.parameters()): p.requires_grad=False
    ds = ThyroidNodeDataset(Path(TRAIN_DIR))
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    prompt="thyroid nodule, ultrasound imaging, clear detail"
    text_emb = text_enc(tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device))[0]
    acc = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    optim = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=1e-2)
    sched = get_scheduler("constant", optimizer=optim, num_warmup_steps=0, num_training_steps=len(dl)*EPOCHS)
    unet, optim, dl, sched = acc.prepare(unet, optim, dl, sched)
    for epoch in range(EPOCHS):
        running=0.0
        for batch in tqdm(dl, desc=f"epoch {epoch+1}/{EPOCHS}"):
            with acc.accumulate(unet):
                imgs, masks = batch["image"].to(device), batch["mask"].to(device)
                lat = vae.encode(imgs).latent_dist.sample()*0.18215
                noise = torch.randn_like(lat)
                t = torch.randint(0, noise_sched.num_train_timesteps, (lat.size(0),), device=device).long()
                lat_noisy = noise_sched.add_noise(lat, noise, t)
                noise_pred = unet(lat_noisy, t, encoder_hidden_states=text_emb).sample
                mask_ds = torch.nn.functional.interpolate(masks, size=lat.shape[-2:], mode="nearest")
                loss = torch.nn.functional.mse_loss(noise_pred*mask_ds, noise*mask_ds)
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(unet.parameters(),1.0)
                optim.step(); sched.step(); optim.zero_grad()
                running+=loss.item()
        print(f"epoch {epoch+1}: loss {running/len(dl):.6f}")
    acc.unwrap_model(unet).save_pretrained("thyroid_node_unet")
if __name__=="__main__":
    finetune_model()
