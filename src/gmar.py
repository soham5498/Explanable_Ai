import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

class GMAR:
    """
    Gradient-weighted Multi-head Attention Rollout (GMAR) implementation for visual explanation
    of transformer-based models like ViT.

    This method computes class-specific heatmaps by combining attention heads using gradients
    as weights and performing residual-aware rollout over layers.

    Args:
        alpha (float): Scaling factor for residual connection during rollout.
        norm_type (str): Normalization method for gradient-based head importance ('l1' or 'l2').
    """
    def __init__(self, alpha=1.0, norm_type='l1'):
        self.alpha = alpha
        self.norm_type = norm_type

    def compute(self, logits, pred_class, attn_weights, model):
        """
        Compute the GMAR heatmap for a given input prediction.

        Args:
            logits (torch.Tensor): Logits from the ViT classifier [1, num_classes].
            pred_class (int): Predicted class index.
            attn_weights (List[torch.Tensor]): Attention matrices from all transformer layers.
            model: The full ViT model (used for gradient backprop).

        Returns:
            torch.Tensor: Normalized class-specific heatmap of shape [14, 14] (or patch grid size).
        """
        model.zero_grad()
        target_logit = logits[0, pred_class]
        target_logit.backward(retain_graph=True)

        weighted_attns = []

        for attn in attn_weights:
            grad = attn.grad
            if self.norm_type == 'l1':
                head_importance = grad.abs().sum(dim=(-1, -2))
            elif self.norm_type == 'l2':
                head_importance = (grad ** 2).sum(dim=(-1, -2)).sqrt()

            head_weights = head_importance / head_importance.sum(dim=-1, keepdim=True)
            head_weights = head_weights.view(1, -1, 1, 1)

            A_weighted = (attn * head_weights).mean(dim=1)
            weighted_attns.append(A_weighted)

        rollout = torch.eye(weighted_attns[0].size(-1)).to(attn_weights[0].device)
        for A in weighted_attns:
            A_residual = A + self.alpha * torch.eye(A.size(-1)).to(A.device)
            A_residual = A_residual / A_residual.sum(dim=-1, keepdim=True)
            rollout = rollout @ A_residual

        cls_influence = rollout[0, 0, 1:]
        side_len = int(cls_influence.numel() ** 0.5)
        cls_map = cls_influence.reshape(side_len, side_len).cpu().detach()
        cls_map = (cls_map - cls_map.min()) / (cls_map.max() - cls_map.min() + 1e-8)
        return cls_map

    def plot_on_image(self, cls_map, original_image, cmap='jet', alpha=0.5):
        """
        Display the GMAR heatmap overlaid on the original image using matplotlib.

        Args:
            cls_map (torch.Tensor): Heatmap of shape [H, W] (typically 14x14).
            original_image (PIL.Image): Original input image.
            cmap (str): Colormap for heatmap.
            alpha (float): Opacity of heatmap overlay.
        """
        heatmap = np.array(cls_map)
        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(original_image.size, resample=Image.BILINEAR)
        heatmap = np.array(heatmap)

        plt.figure(figsize=(8, 8))
        plt.imshow(original_image)
        plt.imshow(heatmap, cmap=cmap, alpha=alpha)
        plt.axis('off')
        plt.show()


    def save_overlay(self, cls_map, original_image, original_image_path, predicted_label=""):
        """
        Save the GMAR heatmap overlay on top of the original image.
        Creates 'gmar_l1/' or 'gmar_l2/' folder depending on norm_type.
        Filename is: originalname_gmar.png
        """
        # Create folder based on norm_type
        output_dir = Path(f"./results/gmar_{self.norm_type}")
        output_dir.mkdir(exist_ok=True)

        # File naming
        image_path = Path(original_image_path)
        base_name = image_path.stem
        output_filename = f"{base_name}_gmar.png"
        output_path = output_dir / output_filename

        # Resize heatmap
        heatmap = np.array(cls_map)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(original_image.size, resample=Image.BILINEAR)
        heatmap_np = np.array(heatmap_img)

        # Plot overlay
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(original_image)
        ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
        ax.set_title(f"GMAR — Predicted: {predicted_label}")
        ax.axis('off')

        # Save to disk
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"GMAR overlay saved: {output_path}")
