import io
import aiohttp
import numpy as np
from PIL import Image
import json


async def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.transpose(2, 0, 1)
    image_array = np.expand_dims(image_array, axis=0)

    async with (aiohttp.ClientSession() as session):
        async with session.post("http://localhost:8000/v2/models/vision_model/infer",
                                json={"inputs": [{"name": "input_image",
                                                  "shape": [1, 3, 224, 224],
                                                  "datatype": "FP32",
                                                  "data": list(image_array.flatten().tolist())}]}
                                ) as resp:
            response = await resp.json()
            embedding = np.array(response['outputs'][0]['data'])
            return embedding


async def process_text(input_text, tokenizer):
    inputs = tokenizer(input_text, return_tensors="np", padding="max_length", truncation=True, max_length=77)

    input_ids = inputs["input_ids"].tolist()
    attention_mask = inputs["attention_mask"].tolist()

    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/v2/models/text_model/infer",
                                json={"inputs": [{"name": "input_ids",
                                                  "shape": [1, 77],
                                                  "datatype": "INT32",
                                                  "data": input_ids},
                                                 {"name": "attention_mask",
                                                  "shape": [1, 77],
                                                  "datatype": "INT32",
                                                  "data": attention_mask[0]}]}
                                ) as resp:
            response = await resp.json()
            embedding = np.array(response['outputs'][0]['data'])
            return embedding
