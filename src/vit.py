import torch
from transformers import ViTImageProcessor, ViTForImageClassification

class CustomViT:
    """
    Custom wrapper around the HuggingFace ViT (Vision Transformer) model,
    enabling access to internal attention weights for explainability purposes.

    Features:
        - Loads a pretrained ViT model.
        - Allows input preprocessing using the matching image processor.
        - Performs a forward pass while capturing attention maps manually.
    """
    def __init__(self, model_name='google/vit-base-patch16-224', device=None):
        """
        Initialize the CustomViT model and processor.

        Args:
            model_name (str): HuggingFace model ID or path to pretrained ViT model.
            device (str): Device to load the model on ('cuda', 'cpu', or None).
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.imagenet_classes = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def preprocess(self, img):
        """
        Preprocess a PIL image into a tensor suitable for ViT input.

        Args:
            img (PIL.Image): Input image.

        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, H, W] on the target device.
        """
        inputs = self.processor(images=img, return_tensors='pt').to(self.device)
        return inputs['pixel_values']

    def forward_with_custom_attention(self, img_tensor):
        """
        Perform a forward pass through ViT while manually extracting per-layer attention maps.

        Args:
            img_tensor (torch.Tensor): Preprocessed image tensor [1, 3, H, W].

        Returns:
            logits (torch.Tensor): Model output logits.
            predicted_class (int): Index of the predicted class.
            class_name (str): Human-readable class label.
            attn_weights (List[torch.Tensor]): List of attention maps per block,
                                               each shaped [1, num_heads, N, N].
        """
        attn_weights = []
        x = self.model.vit.embeddings.patch_embeddings(img_tensor)
        cls_token = self.model.vit.embeddings.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.vit.embeddings.position_embeddings
        x = self.model.vit.embeddings.dropout(x)

        for blk in self.model.vit.encoder.layer:
            B, N, C = x.shape
            norm_x = blk.layernorm_before(x)
            qkv = (
                blk.attention.attention.query(norm_x),
                blk.attention.attention.key(norm_x),
                blk.attention.attention.value(norm_x)
            )
            q, k, v = qkv
            num_heads = blk.attention.attention.num_attention_heads
            head_dim = C // num_heads
            q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, N, num_heads, head_dim).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = attn.softmax(dim=-1)
            attn.retain_grad()
            context = attn @ v
            context = context.transpose(1, 2).reshape(B, N, C)
            attn_out = blk.attention.output.dense(context)
            x = x + attn_out

            mlp_out = blk.intermediate.dense(blk.layernorm_after(x))
            mlp_out = torch.nn.functional.gelu(mlp_out)
            mlp_out = blk.output.dense(mlp_out)
            x = x + mlp_out

            attn_weights.append(attn)

        x = self.model.vit.layernorm(x)
        cls_embedding = x[:, 0]
        logits = self.model.classifier(cls_embedding)
        predicted_class = logits.argmax(dim=-1).item()
        class_name = self.imagenet_classes[predicted_class]

        return logits, predicted_class, class_name, attn_weights
