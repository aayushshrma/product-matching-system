import base64
import json
from PIL import Image
import numpy as np
from metadata_db import db
from vector_db import add_embedding_to_index
from inference import process_image
import asyncio

products = [{"product_id": "001",
             "name": "Brown Boots",
             "price": 3999,
             "category": "Footwear",
             "image_path": "sample/boots.jpg"},
            { "product_id": "002",
              "name": "Blue Hoodie",
              "price": 1499,
              "category": "Apparel",
              "image_path": "sample/hoodie.jpg"},
            {"product_id": "003",
             "name": "Tablet",
             "price": 19999,
             "category": "Electronics",
             "image_path": "sample/tablet.jpg"}]

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
