import base64
import json
from PIL import Image
import numpy as np
from metadata_db import db
from vector_db import add_embedding_to_index
from inference import process_image
import asyncio

products = [{"product_id": "001",
             "name": "Red Sneakers",
             "price": 1499,
             "category": "Footwear",
             "image_path": "app/sample/red_shoe.jpg"},
            { "product_id": "002",
              "name": "Black Hoodie",
              "price": 1299,
              "category": "Apparel",
              "image_path": "app/sample/black_hoodie.jpg"}]

async def populate():
    for product in products:
        with open(product['image_path'], 'rb') as f:
            image_bytes = f.read()
            emb = await process_image(image_bytes)
            await add_embedding_to_index(product['product_id'], emb)

        db.catalog.insert_one({"product_id": product['product_id'],
                               "name": product['name'],
                               "price": product['price'],
                               "category": product['category'],
                               "image_url": product['image_path']})


if __name__ == "__main__":
    asyncio.run(populate())
