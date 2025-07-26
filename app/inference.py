import aiohttp
import numpy as np
import json


async def process_image(image_bytes):
    async with (aiohttp.ClientSession() as session):
        async with session.post("http://triton:8000/v2/models/vision_model/infer",
                                json={"inputs": [{"name": "input_image",
                                                  "shape": [1, 3, 224, 224],
                                                  "datatype": "UINT8",
                                                  "data": list(image_bytes[:150528])}]}
                                ) as resp:
            response = await resp.json()
            embedding = np.array(response['outputs'][0]['data'])
            return embedding
