# Generative Adversarial Networks (GANs)

This project explores image generation with **Vanilla GANs** and **Deep
Convolutional GANs (DCGANs)** using PyTorch and TensorFlow. The examples use
the Food-101 image dataset.

## Project contents

| File | Description |
| --- | --- |
| `gan_comparison.py` | Trains and compares a Vanilla GAN and a DCGAN using PyTorch. |
| `Vanilla_GAN.ipynb` | Notebook implementation of a Vanilla GAN in PyTorch. |
| `DCGAN.ipynb` | Notebook implementation of a DCGAN in PyTorch. |
| `GAN.ipynb` | PyTorch experiment comparing GANs on selected Food-101 classes. |
| `DCGAN-genai.ipynb` | TensorFlow GAN experiments, intended for Google Colab. |

## Requirements

- Python with versions supported by the frameworks listed in `requirements.txt`
- Food-101 data for the PyTorch script and PyTorch notebooks
- A Kaggle account/token for the Colab notebook; GPU access is recommended for
  practical training times

Install the dependencies in a virtual environment (Windows PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, follow the installation selector at
[pytorch.org](https://pytorch.org/get-started/locally/) and choose the build
matching your CUDA setup. The models can also run on CPU, but training will be
much slower.

## Food-101 data

Dataset files are intentionally not stored in this repository. Download Food-101
from the [official dataset page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
or the source configured in a notebook, then extract it so the image directory
contains class folders, for example:

```text
dataset/food-101/food-101/images/apple_pie/*.jpg
dataset/food-101/food-101/images/bibimbap/*.jpg
```

The PyTorch script and image-folder notebooks expect the images at
`dataset/food-101/food-101/images`. If the extracted archive has a different
folder structure, update `image_dir` in the notebook you are using.

`GAN.ipynb` downloads Food-101 with `kagglehub`. `DCGAN-genai.ipynb` is designed
for Google Colab and uses the Colab Secrets facility for the Kaggle API token.
Create a Colab secret named `KAGGLE_API_TOKEN` and grant the notebook access to
it. Do not put API tokens or other credentials in notebook cells or commit them
to Git.

## Run the PyTorch comparison

From the repository root, with the dataset in the layout above:

```powershell
python gan_comparison.py
```

The script uses five epochs by default. Adjust its configuration values to
change the training duration, batch size, or model dimensions. The notebooks
contain their own training configuration and can be opened in Jupyter or VS
Code.

## Dataset citation

Food-101 is described in:

> Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. "Food-101 - Mining
> Discriminative Components with Random Forests." ECCV 2014.

See the [Food-101 dataset page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
for dataset details and terms of use.

## License

No software license is currently specified for this repository.
