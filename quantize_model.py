import torch
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                  resume_download=True)
model.eval()
print("Model Downloaded!")

dummy_image = torch.randn(1, 3, 224, 224)
torch.onnx.export(model.vision_model,
                  dummy_image,
                  "model.onnx",
                  input_names=["input_image"],
                  output_names=["embedding"],
                  dynamic_axes={"input_image": {0: "batch"}},
                  opset_version=14)

dummy_ids = torch.randint(0, 10000, (1,77))
dummy_mask = torch.ones((1,77), dtype=torch.int64)

torch.onnx.export(model.text_model,
                  (dummy_ids, dummy_mask),
                  "model.onnx",
                  input_names=["input_ids", "attention_mask"],
                  output_names=["embedding"],
                  dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
                  opset_version=14)

print("Export Complete!")