# Computer Vision Deep Labs

Selected Master DS **Computer Vision** labs (code/notebooks only — no multi-GB optical-flow dumps).

## Labs included

| Folder | Topic |
|--------|--------|
| `lab2-keypoints/` | Keypoints / local features |
| `lab5/` | Optical-flow style networks (synthetic OF + CNN) |
| `lab6/` | Unsupervised feature learning (codebooks / autoencoders on CIFAR-style data) |

## Stack

PyTorch / TensorFlow-Keras (per notebook) · NumPy · Matplotlib · OpenCV-style CV utilities

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook lab5/sekou.ipynb
```

Datasets are downloaded or regenerated inside the notebooks — **not** shipped here.

## What was deliberately excluded

- `lab4` optical-flow `.pkl` dumps (~tens of GB)
- Large `.h5` / CIFAR batch binaries

## Why publish

Shows CV coursework depth beyond YOLO fine-tuning: geometric features, flow, unsupervised representation learning.
