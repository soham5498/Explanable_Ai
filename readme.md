# 🧠 ViT Explainability: LeGrad, GMAR, and Classical Rollout

This project implements **visual explanations** for Vision Transformers (ViTs) using three methods:

* 🎯 **LeGrad** —  An Explainability Method for Vision Transformers via Feature Formation Sensitivity
* 🔍 **GMAR** — Gradient-weighted Multi-head Attention Rollout
* 🧱 **Classical Rollout** — Layer-wise attention rollout without gradients

All explanations are visualized as **attention heatmaps** overlaid on original input images.
The goal is to analyze **which patches the model attends to** when making predictions.

---

## 📂 Project Structure

```
project/
├ images/                # Input images (supports .jpg/.png/.jpeg)
├ results/               # Output folder for visualizations
│  ├ legrad/             # Output heatmaps for LeGrad
│  ├ gmar/               # Output heatmaps for GMAR (with prefix: l1_ or l2_)
│  └ rollout/            # Output heatmaps for classical attention rollout
├ src/                   # Source code directory
│  ├ gmar.py             # GMAR explanation method
│  ├ legrad.py           # LeGrad explanation method
│  ├ rollout.py          # Classical attention rollout
│  ├ vit.py              # Custom ViT wrapper with attention extraction
│  ├ main.py             # Main driver for running explanations
│  └ __pycache__/        # Python bytecode cache
├ __pycache__/
└ readme.md              # This file
```

---

## 🚀 How to Use

### 1. Place your input images

Put images in the `images/` folder. Supported formats: `.jpg`, `.jpeg`, `.png`

### 2. Run the project

Navigate to the `src/` directory and run:

```bash
cd src
python main.py
```

You'll be prompted to choose an explanation method:

```bash
Which method? (legrad/gmar/rollout):
```

* `legrad`: Gradient-weighted patch sensitivity
* `gmar`: GMAR with choice of `l1` or `l2` norm
* `rollout`: Classical attention rollout (Abnar & Zuidema, 2020)

---

## 🗸️ Example Output

Each method saves a heatmap overlay in its respective subfolder under `results/`:

```
results/
├ legrad/cat_legrad.png
├ gmar/l1_cat_gmar.png
├ gmar/l2_dog_gmar.png
├ rollout/cat_rollout.png
```

Titles on the images show the predicted class from the model.

---

## ⚙️ Model Used

* `google/vit-base-patch16-224` from Hugging Face
* Pretrained on ImageNet-1K
* Predictions and attention weights extracted using `transformers` library

---

## 📦 Requirements

Install dependencies via pip:

```bash
pip install torch torchvision transformers matplotlib pillow numpy
```

---

## 📚 References

* [LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](arxiv.org/pdf/2404.03214)
* [Quantifying Attention Flow in Transformers](https://arxiv.org/pdf/2005.00928)
* [GMAR: Gradient-weighted Multi-head Attention Rollout (ICCV 2021)](https://arxiv.org/pdf/2504.19414)

---

## ✨ Author

Prepared by \Soham Joita
Master’s student @ OTH Amberg-Weiden
AI for Industrial Applications 💡
