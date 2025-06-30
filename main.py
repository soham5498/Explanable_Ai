
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from src.vit import CustomViT
from src.legrad import LeGradExplainer
from src.rollout import AttentionRollout
from src.gmar import GMAR
from src.metrics import compute_pixel_accuracy, compute_iou, compute_ap

# === Ask user for method choice ===
choice = input("Which method? (legrad/gmar/rollout): ").strip().lower()

# === Initialize model and directories ===
vit = CustomViT()
image_dir = Path('./images')
mask_dir = Path('./masks')

# Match jpg, jpeg, png
image_files = list(image_dir.glob('*.jpg')) \
            + list(image_dir.glob('*.jpeg')) \
            + list(image_dir.glob('*.png'))

# === Metric summaries ===
pixel_acc_list, iou_list, ap_list = [], [], []

# === Check if images found ===
for image_file in image_files:
    img = Image.open(image_file).convert('RGB')
    img_tensor = vit.preprocess(img)
    logits, pred_idx, class_name, attn_weights = vit.forward_with_custom_attention(img_tensor)
    print(f"\n Processing: {image_file.name} → {class_name}")

    # === Choice of legrad, gmar, or rollout ===
    if choice == "legrad":
        explainer = LeGradExplainer(attn_weights, logits, pred_idx, predicted_label=class_name)
        explainer.compute_layer_maps()
        heatmap = explainer.merge_heatmap()
        explainer.save_overlay(img, heatmap, original_image_path=image_file)
        side_len = int(heatmap.shape[0] ** 0.5)
        heatmap_2d = heatmap.reshape(side_len, side_len)
        final_map = torch.tensor(heatmap_2d)

    elif choice == "gmar":
        explainer = GMAR(alpha=1.0, norm_type='l1')
        cls_map = explainer.compute(logits, pred_idx, attn_weights, model=vit.model)
        explainer.save_overlay(cls_map, img, original_image_path=image_file, predicted_label=class_name)
        final_map = cls_map

    elif choice == "rollout":
        explainer = AttentionRollout(add_residual=True)
        cls_map = explainer.compute_rollout(attn_weights)
        explainer.save_overlay(cls_map, img, original_image_path=image_file, predicted_label=class_name)
        final_map = cls_map

    else:
        print(" Invalid choice. Use 'legrad' or 'gmar' or 'rollout'.")
        break

    # === Load matching mask (with .png extension) ===
    mask_file = Path("./masks") / (image_file.stem + ".png")
    if not mask_file.exists():
        print(f" Mask not found for {image_file.name}")
        continue
        
    gt_mask = Image.open(mask_file).convert('L').resize(final_map.shape[::-1])  # (W, H)
    gt_mask = torch.tensor(np.array(gt_mask) > 127).int()

    # === Binarize prediction map ===
    pred_mask = (final_map > 0.5).int()

    # === Compute metrics ===
    pixel_acc = compute_pixel_accuracy(pred_mask, gt_mask)
    iou = compute_iou(pred_mask, gt_mask)
    ap = compute_ap(final_map, gt_mask)

    print(f" PixelAcc: {pixel_acc:.4f} | IoU: {iou:.4f} | AP: {ap:.4f}")

    pixel_acc_list.append(pixel_acc)
    iou_list.append(iou)
    ap_list.append(ap)

# === Summary ===
print("\n=== Final Average Metrics ===")
if pixel_acc_list:
    print(f"Mean PixelAcc: {sum(pixel_acc_list)/len(pixel_acc_list):.4f}")
    print(f"Mean IoU: {sum(iou_list)/len(iou_list):.4f}")
    print(f"Mean AP: {sum(ap_list)/len(ap_list):.4f}")
else:
    print("No valid masks found. Metrics not computed.")

print("\n All images done.")

