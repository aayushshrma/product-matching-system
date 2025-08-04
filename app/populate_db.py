from metadata_db import db
from vector_db import add_embedding_to_index
from inference import process_image
from inference import process_text


products = [{"product_id": "001",
             "name": "Brown Boots",
             "price": 3999,
             "category": "Footwear",
             "image_path": "catalog/boots.jpg"},
            { "product_id": "002",
              "name": "Blue Hoodie",
              "price": 1499,
              "category": "Apparel",
              "image_path": "catalog/hoodie.jpg"},
            {"product_id": "003",
             "name": "Tablet",
             "price": 19999,
             "category": "Electronics",
             "image_path": "catalog/tablet.jpg"},
            {"product_id": "004",
             "name": "Table",
             "price": 14999,
             "category": "Furniture",
             "image_path": "catalog/table.jpg"},
            {"product_id": "005",
             "name": "Phone",
             "price": 49999,
             "category": "Electronics",
             "image_path": "catalog/phone.jpg"}
            ]

async def populate():
    for product in products:
        with open(product['image_path'], 'rb') as f:
            image_bytes = f.read()
        image_emb = await process_image(image_bytes=image_bytes)
        text_emb = await process_text(input_text=product["name"])
        await add_embedding_to_index(product_id=product['product_id'], image_emb=image_emb,
                                     text_emb=text_emb)

        db.catalog.insert_one({"product_id": product['product_id'],
                               "name": product['name'],
                               "price": product['price'],
                               "category": product['category'],
                               "image_url": product['image_path']})

