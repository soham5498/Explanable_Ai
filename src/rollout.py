import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

class AttentionRollout:
    def __init__(self, add_residual=True):
        self.add_residual = add_residual

    def compute_rollout(self, attn_weights):
        """
        Classical Attention Rollout from a list of attention matrices.

        Args:
            attn_weights: List of attention maps per layer [B, H, N, N]
        
        Returns:
            cls_map: 2D torch.Tensor, normalized heatmap [14, 14]
        """
        device = attn_weights[0].device
        rollout = torch.eye(attn_weights[0].size(-1)).to(device)

        for A in attn_weights:
            A_mean = A.mean(dim=1)  # average over heads [B, N, N]
            if self.add_residual:
                A_res = A_mean + torch.eye(A_mean.size(-1)).to(device)
                A_res = A_res / A_res.sum(dim=-1, keepdim=True)
            else:
                A_res = A_mean
            rollout = rollout @ A_res

        # Get [CLS] token influence on patches
        cls_influence = rollout[0, 0, 1:]  # skip [CLS] itself
        side_len = int(cls_influence.numel() ** 0.5)
        cls_map = cls_influence.reshape(side_len, side_len).cpu().detach()
        cls_map = (cls_map - cls_map.min()) / (cls_map.max() - cls_map.min() + 1e-8)
        return cls_map

    def save_overlay(self, cls_map, image, original_image_path, predicted_label=""):
        """
        Saves the heatmap overlay image with attention rollout.
        """
        output_dir = Path("./results/rollout")
        output_dir.mkdir(exist_ok=True)

        image_path = Path(original_image_path)
        base_name = image_path.stem
        output_path = output_dir / f"{base_name}_rollout.png"

        heatmap = np.array(cls_map)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(image.size, resample=Image.BILINEAR)
        heatmap_np = np.array(heatmap_img)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)
        ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
        ax.set_title(f"Rollout — Predicted: {predicted_label}")
        ax.axis('off')

        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"✅ Rollout overlay saved: {output_path}")
