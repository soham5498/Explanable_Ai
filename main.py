from pathlib import Path
from PIL import Image
from src.vit import CustomViT
from src.legrad import LeGradExplainer
from src.rollout import AttentionRollout
from src.gmar import GMAR

# === Ask once:
choice = input("Which method? (legrad/gmar/rollout): ").strip().lower()

vit = CustomViT()
image_dir = Path('./images')

# Match jpg, jpeg, png
image_files = list(image_dir.glob('*.jpg')) \
            + list(image_dir.glob('*.jpeg')) \
            + list(image_dir.glob('*.png'))

for image_file in image_files:
    img = Image.open(image_file).convert('RGB')
    img_tensor = vit.preprocess(img)
    logits, pred_idx, class_name, attn_weights = vit.forward_with_custom_attention(img_tensor)
    print(f"\n Processing: {image_file.name} → {class_name}")

    if choice == "legrad":
        legrad = LeGradExplainer(attn_weights, logits, pred_idx, predicted_label=class_name)
        legrad.compute_layer_maps()
        heatmap = legrad.merge_heatmap()
        legrad.save_overlay(img, heatmap, original_image_path=image_file)

    elif choice == "gmar":
        gmar = GMAR(alpha=1.0, norm_type='l1')
        cls_map = gmar.compute(logits, pred_idx, attn_weights, model=vit.model)
        gmar.save_overlay(cls_map, img, original_image_path=image_file, predicted_label=class_name)
    
    elif choice == "rollout":
        rollout = AttentionRollout(add_residual=True)
        cls_map = rollout.compute_rollout(attn_weights)
        rollout.save_overlay(cls_map, img, original_image_path=image_file, predicted_label=class_name)



    else:
        print(" Invalid choice. Use 'legrad' or 'gmar' or 'rollout'.")
        break

print("\n✅ All images done.")

