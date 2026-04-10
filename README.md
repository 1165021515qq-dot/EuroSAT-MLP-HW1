# EuroSAT-MLP-HW1

HW1: Build a three-layer MLP from scratch for EuroSAT image classification.

## Project Overview
This project implements a three-layer MLP classifier from scratch using NumPy for land cover image classification on the EuroSAT RGB dataset.

The implementation does not use PyTorch, TensorFlow, JAX, or any automatic differentiation framework. All forward propagation, backpropagation, gradient computation, and parameter updates are implemented manually.

## Dataset
The dataset used in this project is EuroSAT RGB, which contains 10 land cover categories:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

The EuroSAT_RGB dataset is not uploaded to this repository because of its large size.

## Environment
Recommended Python version: 3.10+

Required packages:

```bash
pip install numpy pillow matplotlib
```

## File Structure
```text
EuroSAT-MLP-HW1/
├── hw1_eurosat_mlp.py
└── README.md
```

## Model Description
The model is a three-layer MLP for multi-class image classification. The implementation supports:

- Custom hidden dimension
- Multiple activation functions: ReLU, Sigmoid, and Tanh
- Cross-entropy loss
- L2 regularization
- SGD optimization
- Learning rate decay
- Validation-based best model saving
- Hyperparameter search

## How to Train
Example command:

```bash
python hw1_eurosat_mlp.py --mode train --data_root "path_to_EuroSAT_RGB" --output_dir "outputs_try6" --image_size 32 --hidden_dim 384 --activation relu --lr 0.02 --lr_decay 0.99 --weight_decay 0.001 --epochs 35 --batch_size 128
```

## How to Search Hyperparameters
Example command:

```bash
python hw1_eurosat_mlp.py --mode search --data_root "path_to_EuroSAT_RGB" --output_dir "outputs_search" --image_size 32 --epochs 20 --batch_size 128 --lr_decay 0.99 --search_lrs 0.02,0.03,0.05 --search_hiddens 256,384 --search_wds 1e-4,5e-4,1e-3 --search_activations relu
```

## How to Test
Example command:

```bash
python hw1_eurosat_mlp.py --mode test --data_root "path_to_EuroSAT_RGB" --weights "path_to_best_model.npz" --output_dir "outputs_test" --image_size 32 --hidden_dim 384 --activation relu
```

## Final Result
Several experiments were conducted to tune the hyperparameters. The best final setting used:

- Hidden dimension: 384
- Activation: ReLU
- Learning rate: 0.02
- Learning rate decay: 0.99
- Weight decay: 0.001
- Epochs: 35
- Batch size: 128

Final performance:

- Best validation accuracy: 0.6659
- Test accuracy: 0.6770

## Model Weights
The trained model weights can be downloaded here:

https://drive.google.com/file/d/1FzyIqAj3jEGxGgHF_Qq_QJbTVG6Qf7Cl/view?usp=drive_link

## GitHub Repository
Project repository link:

https://github.com/1165021515qq-dot/EuroSAT-MLP-HW1

## Notes
This repository only contains the source code and documentation.

The dataset is not included because of its large size.

The trained model weights are shared through Google Drive.
