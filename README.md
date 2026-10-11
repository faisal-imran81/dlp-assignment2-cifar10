# DL Assignment 2 — CIFAR-10 Robust Classification

**Student:** Faisal Imran  
**Roll No:** 23F-0563  
**Course:** Deep Learning for Perception (Fall 2026)  
**University:** FAST NUCES, Chiniot-Faisalabad Campus  

---

## Problem
Robust image classification on CIFAR-10 with only 500 training images
against a shifted (corrupted) test set.

## Architecture
- Custom CNN with mandatory Inception-style module
- Total parameters: 46,674 (within 1M limit)
- No pretrained weights

## Results

| Configuration       | Clean Test     | Shifted Test   |
|---------------------|----------------|----------------|
| No Augmentation     | 0.4395 ± 0.005 | 0.2224 ± 0.013 |
| Geometric Only      | 0.3867 ± 0.003 | 0.2457 ± 0.003 |
| Photometric Only    | 0.4397 ± 0.002 | 0.2561 ± 0.012 |
| Geo + Photo + Cutout| 0.4076 ± 0.009 | 0.2606 ± 0.009 |
| **Final Model**     | **0.4076**     | **0.2606**     |

## Reproduce

1. Open `DL_ASS02_23F_0563.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Runtime → Run all
4. Fixed seed: `SEED = 42`

## Dependencies

torch
torchvision
numpy
matplotlib
sklearn


## Key Findings
- Photometric augmentation gave biggest robustness gain
- TTA gave negligible improvement on small model
- Overfitting main challenge with 400 training images
