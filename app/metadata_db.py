from motor.motor_asyncio import AsyncIOMotorClient
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client.products

async def get_product_metadata(product_id):
    result = db.catalog.find_one({"product_id": product_id}, {"_id": 0})
    return await result