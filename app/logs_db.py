from motor.motor_asyncio import AsyncIOMotorClient
import datetime
import base64
import os

client = AsyncIOMotorClient(os.environ["MONGO_URI"])
db = client.logs

async def log_query(image_bytes, top_match_id, error=None):
    log_entry = {"timestamp": datetime.datetime.now(),
                 "input": base64.b64encode(image_bytes).decode('utf-8'),
                 "top_match_id": top_match_id,
                 "error": error}

    await db.queries.insert_one(log_entry)