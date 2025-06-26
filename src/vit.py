import torch
from transformers import ViTImageProcessor, ViTForImageClassification

class CustomViT:
    def __init__(self, model_name='google/vit-base-patch16-224', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.imagenet_classes = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def preprocess(self, img):
        inputs = self.processor(images=img, return_tensors='pt').to(self.device)
        return inputs['pixel_values']

    def forward_with_custom_attention(self, img_tensor):
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
