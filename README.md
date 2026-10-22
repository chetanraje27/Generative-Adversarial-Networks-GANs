# Generative Adversarial Networks (GANs)

This project compares **Vanilla GANs** and **Deep Convolutional GANs (DCGANs)**
for food-image generation with PyTorch and the Food-101 dataset.

## Main project

[`GAN.ipynb`](./GAN.ipynb) is the primary project file. It downloads Food-101
with `kagglehub`, selects 1,000 images from each of five classes (pizza, sushi,
donuts, hamburgers, and ice cream), trains both GAN architectures, plots
training progress, and saves the trained generator weights.

Open `GAN.ipynb` in Google Colab or Jupyter and run its cells in order. A
GPU-enabled runtime is recommended: the notebook trains both models for up to
300 epochs, which can take a long time on CPU. The notebook downloads the data
automatically; no local Food-101 folder setup is needed for this workflow.

## Project contents

| File | Description |
| --- | --- |
| `GAN.ipynb` | **Main project:** trains and compares Vanilla GAN and DCGAN on selected Food-101 classes using PyTorch. |
| `gan_comparison.py` | Standalone PyTorch comparison script; expects Food-101 images in the local folder described below. |
| `Vanilla_GAN.ipynb` | Separate PyTorch Vanilla GAN example. |
| `DCGAN.ipynb` | Separate PyTorch DCGAN example. |
| `DCGAN-genai.ipynb` | Additional TensorFlow GAN experiments, intended for Google Colab. |

## Requirements

The main notebook installs `torchmetrics`, `torch-fidelity`, and `kagglehub`
in its first cell. For local Jupyter use, install the project dependencies in a
virtual environment (Windows PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

For CUDA-enabled PyTorch, follow the installation selector at
[pytorch.org](https://pytorch.org/get-started/locally/) and choose the build
matching your CUDA setup. The main notebook can download the public dataset
through `kagglehub`; if Kaggle requests authentication, configure credentials
using Kaggle's recommended method and never commit them to this repository.

## Running the standalone script

The standalone `gan_comparison.py` script and the two separate image-folder
notebooks expect Food-101 to be downloaded and extracted locally. The expected
image directory is `dataset/food-101/food-101/images`, containing class
subfolders. Dataset files are not included in this repository.

Run the script from the repository root after placing the dataset in that
location:

```powershell
python gan_comparison.py
```

The script uses five epochs by default. Adjust its configuration values to
change the training duration, batch size, or model dimensions.

## Dataset citation

Food-101 is described in:

> Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. "Food-101 - Mining
> Discriminative Components with Random Forests." ECCV 2014.

See the [Food-101 dataset page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
for dataset details and terms of use.

## License

No software license is currently specified for this repository.
