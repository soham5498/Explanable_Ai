import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

class LeGradExplainer:
    def __init__(self, attn_weights, logits, predicted_class, predicted_label=""):
        self.attn_weights = attn_weights
        self.logits = logits
        self.predicted_class = predicted_class
        self.predicted_label = predicted_label
        self.layer_maps = []

    def compute_layer_maps(self):
        c = self.logits[0, self.predicted_class]
        self.logits.grad = None
        c.backward()
        layer_maps = []

        for idx, A in enumerate(self.attn_weights):
            grad = A.grad
            grad_pos = grad.clamp(min=0)
            gcam = grad_pos * A
            gcam_mean = gcam.mean(dim=1)
            gcam_no_cls = gcam_mean[:, 1:, 1:]
            patch_score = gcam_no_cls.mean(dim=-1)
            layer_maps.append(patch_score)
            # print(f"Block {idx+1} done: patch_score shape {patch_score.shape}")

        self.layer_maps = layer_maps
        return layer_maps

    def merge_heatmap(self):
        merged = torch.stack(self.layer_maps).mean(dim=0)
        heatmap = merged.detach().squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        print(f"✅ Final LeGrad patch heatmap shape: {heatmap.shape}")
        return heatmap
        # merged = torch.stack(self.layer_maps).mean(dim=0)
        # heatmap = merged.detach().squeeze().cpu().numpy()
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # side_len = int(heatmap.shape[0] ** 0.5)
        # heatmap = heatmap.reshape(side_len, side_len)

        # print(f"✅ Final LeGrad patch heatmap shape: {heatmap.shape}")
        # return heatmap


    def visualize_overlay(self, img, heatmap):
        img_np = np.array(img.resize((224, 224))) / 255.0
        heatmap_grid = heatmap.reshape(14, 14)
        heatmap_tensor = torch.tensor(heatmap_grid).unsqueeze(0).unsqueeze(0)
        heatmap_upsampled = F.interpolate(heatmap_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        heatmap_upsampled = heatmap_upsampled.squeeze().cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.imshow(heatmap_upsampled, cmap='jet', alpha=0.5)
        plt.title(f"LeGrad — Predicted: {self.predicted_label}")

        plt.axis('off')
        plt.show(block=True)
        
    def save_overlay(self, img, heatmap, original_image_path):
        """
        Save the heatmap overlay on top of the original image to disk.
        Creates 'legrad/' folder if it doesn't exist.
        The saved filename is: originalname_legrad.png
        Args:
            img: PIL.Image original input image
            heatmap: 2D numpy or tensor (patch-level)
            original_image_path: str, path to the original image (used to name output file)
        """
        output_dir = Path("./results/legrad")
        output_dir.mkdir(exist_ok=True)

        # Accept str or Path safely
        image_path = Path(original_image_path)
        base_name = image_path.stem  
        output_path = output_dir / f"{base_name}_legrad.png"

        # === Derive output filename ===
        base_name = os.path.basename(original_image_path)
        name_without_ext, _ = os.path.splitext(base_name)
        output_filename = f"{name_without_ext}_legrad.png"
        output_path = os.path.join(output_dir, output_filename)

        
        # === Prepare overlay ===
        img_resized = img.resize((224, 224))
        img_np = np.array(img_resized) / 255.0

        heatmap_grid = heatmap.reshape(14, 14)
        heatmap_tensor = torch.tensor(heatmap_grid).unsqueeze(0).unsqueeze(0)
        heatmap_upsampled = F.interpolate(heatmap_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        heatmap_upsampled = heatmap_upsampled.squeeze().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_np)
        ax.imshow(heatmap_upsampled, cmap='jet', alpha=0.5)
        ax.set_title(f"LeGrad — Predicted: {self.predicted_label}")

        ax.axis('off')

        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"✅ Saved overlay: {output_path}")