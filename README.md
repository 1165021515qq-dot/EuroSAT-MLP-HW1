# EuroSAT-MLP-HW1

HW1: Build a three-layer MLP from scratch for EuroSAT image classification.

## Project Overview
This project implements a three-layer MLP classifier from scratch using NumPy for land cover image classification on the EuroSAT RGB dataset.  
The implementation does not use PyTorch, TensorFlow, JAX, or any automatic differentiation framework.

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

## Environment
Recommended Python version: 3.10+

Required packages:

```bash
pip install numpy pillow matplotlib
