import torch
import torch.nn as nn
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                  resume_download=True)
model.eval()
print("Model Downloaded!")


class CLIPImageEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, image):
        return self.clip_model.get_image_features(image)


class CLIPTextEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, input_ids, attention_mask):
        return self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)


dummy_image = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    vis_output = model.get_image_features(dummy_image)
print(f"VISION MODEL OUTPUT SHAPE: {vis_output.shape}")

# --- Image Model ---
image_exporter = CLIPImageEncoderWrapper(model)
torch.onnx.export(image_exporter,
                  (dummy_image,),
                  "clip_image_encoder.onnx",
                  input_names=["input_image"],
                  output_names=["embedding"],
                  dynamic_axes={"input_image": {0: "batch"}},
                  opset_version=14)

dummy_ids = torch.randint(0, 10000, (1,77))
dummy_mask = torch.ones((1,77), dtype=torch.int64)

with torch.no_grad():
    text_output = model.get_text_features(input_ids=dummy_ids, attention_mask=dummy_mask)
print(f"TEXT MODEL OUTPUT SHAPE: {text_output.shape}")

# --- Text Model ---
text_exporter = CLIPTextEncoderWrapper(model)
torch.onnx.export(text_exporter,
                  (dummy_ids, dummy_mask),
                  "clip_text_encoder.onnx",
                  input_names=["input_ids", "attention_mask"],
                  output_names=["embedding"],
                  dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
                  opset_version=14)

print("ONNX Export Complete ✅")