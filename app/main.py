from fastapi import FastAPI, File, UploadFile
from inference import process_image
from vector_db import search_nearest_embeddings
from metadata_db import get_product_metadata
from logs_db import log_query
from fastapi.responses import JSONResponse
import base64


app = FastAPI

@app.post("/match")
async def match_product(file: UploadFile = File()):
    image_bytes = await file.read()
    try:
        embedding = await process_image(image_bytes=image_bytes)
        nearest_match = await search_nearest_embeddings(query_embedding=embedding)
        metadata = [await get_product_metadata(n["product_id"]) for n in nearest_match]
        await log_query(image_bytes=image_bytes, top_match_id=nearest_match[0]["product_id"])
        return JSONResponse(content={"matches": metadata})
    except Exception as e:
        await log_query(image_bytes=image_bytes, top_match_id=None, error=str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

