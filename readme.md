# 🧠 Vision Transformer Explainability: LeGrad Vs GMAR vs Classical Rollout

This project provides a unified framework for **interpreting Vision Transformers (ViTs)** using three state-of-the-art explanation methods:

- 🎯 **LeGrad** — An Explainability Method for Vision Transformers via Feature Formation Sensitivity
- 🔍 **GMAR** — Gradient-weighted Multi-head Attention Rollout with L1 and L2 norms
- 🧱 **Classical Rollout** — Recursive layer-wise attention propagation (Abnar & Zuidema, 2020)

These methods generate **visual attention heatmaps** to understand **which regions the model focuses on** during prediction. The project also supports **quantitative evaluation** using segmentation masks.

---

## 📂 Project Structure

```
project/
├ images/                # Input images for classification and visualization
├ masks/                # Ground truth binary masks for evaluation (same name as input image, .png format)
├ results/               # Output attention heatmaps
│  ├ legrad/             # LeGrad overlays
│  ├ gmar_l1/            # GMAR (L1 norm) overlays
│  ├ gmar_l2/            # GMAR (L2 norm) overlays
│  └ rollout/            # Classical Rollout overlays
├ src/                   # Source files
│  ├ vit.py              # Custom ViT wrapper with attention extraction
│  ├ legrad.py           # LeGrad method implementation
│  ├ gmar.py             # GMAR method implementation
│  ├ rollout.py          # Classical rollout implementation
│  ├ metrics.py          # PixelAcc, IoU, and mAP metrics
│  └ __pycache__/        # Python cache
├ main.py                # Run visualization and evaluation
├ readme.md              # This file
```

---

## 🚀 How to Use

### 1. Place input images

- Put images into the `images/` folder.
- Add binary masks (same base filename, `.png`) into the `masks/` folder.

### 2. Run the project

```bash
python main.py
```

Choose one of the methods when prompted:

```bash
Which method? (legrad/gmar/rollout):
```

### 3. View and evaluate results

- Heatmaps will be saved in `results/<method>/`
- If masks are present, the script automatically computes:
  - **Pixel Accuracy**
  - **Mean Intersection over Union (mIoU)**
  - **Mean Average Precision (mAP)**

---

## 📊 Quantitative Evaluation

The project evaluates interpretability performance using three metrics:

- **Pixel Accuracy**: Proportion of pixels correctly classified
- **mIoU**: Intersection over Union between predicted and ground-truth masks
- **mAP**: Pixel-wise mean Average Precision

These are computed after binarizing the heatmap using a **threshold of 0.5**.

---

## 🧪 Example Outputs

**Heatmaps:**

```
results/
├ legrad/2007_000720_legrad.png
├ gmar_l1/2007_000720_gmar.png
├ gmar_l2/2007_000720_gmar.png
├ rollout/2007_000720_rollout.png
```

**Metrics Output:**
```
PixelAcc: 0.88 | IoU: 0.53 | AP: 0.85
```

---

## ⚙️ Model Details

- Base model: `google/vit-base-patch16-224` (from Hugging Face)
- Pretrained on ImageNet-1K
- No fine-tuning; all explanations use inference-only

---

## 📦 Installation

Install dependencies:

```bash
pip install torch torchvision transformers matplotlib pillow numpy scikit-learn
```

---

## 📚 References

- [LeGrad: An Explainability Method for Vision Transformers (2024)](https://arxiv.org/pdf/2404.03214)
- [GMAR: Gradient-weighted Multi-head Attention Rollout (2025)](https://arxiv.org/pdf/2504.19414)
- [Quantifying Attention Flow in Transformers (2020)](https://arxiv.org/pdf/2005.00928)

---

## ✨ Author

Prepared by **Soham Joita**  
Master’s student, OTH Amberg-Weiden  
AI for Industrial Applications 💡
