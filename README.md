
### Thyroid Nodule Ultrasound In-painting Pipeline

This mini-package provides a reproducible workflow for

1. **Dataset statistics** – compute dataset-specific mean and std for grayscale normalisation.
2. **Fine-tuning** a Stable-Diffusion v1-4 U-Net on masked ultrasound patches.
3. **In-painting** new synthetic nodules with a smooth transition and subtle edge emphasis.

```bash
# 0.  Install dependencies
pip install -r requirements.txt

# 0a. (Optional) Pre-download the Stable-Diffusion v1-4 weights
python get_model.py            # or: huggingface-cli download CompVis/stable-diffusion-v1-4

# 1.  Verify / update dataset statistics
python stats.py  --data-dir "data/data_train_resize_512"

# 2.  Fine-tune the model
python train.py   --epochs 4 --batch-size 1

# 3.  Generate synthetic images
python inpaint.py --start-index 162
```

Folder structure is controlled centrally in **`config.py`** – adjust only there.

You can access and download the datasets from the following external sources:

TDID dataset: URL: http://cimalab.unal.edu.co/applications/thyroid/

TN3K dataset: URL: https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view

Thyroid Digital Image Database (TDID) dataset: URL: https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5
