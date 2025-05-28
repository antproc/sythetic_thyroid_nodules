
"""Download Stable-Diffusion v1-4 weights to the local cache."""
from diffusers import StableDiffusionPipeline
from config import MODEL_ID, CACHE_DIR
def main():
    StableDiffusionPipeline.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=False)
    print(f"Model cached under {(CACHE_DIR/ MODEL_ID.replace('/', '-'))}")
if __name__=="__main__":
    main()
