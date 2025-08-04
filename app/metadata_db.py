from motor.motor_asyncio import AsyncIOMotorClient
import os

client = AsyncIOMotorClient(os.environ["MONGO_URI"])

db = client.products

async def get_product_metadata(product_id):
    result = db.catalog.find_one({"product_id": product_id}, {"_id": 0})
    return await result

async def clear_catalog():
    await db.catalog.delete_many({})