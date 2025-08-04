import os
import asyncio
import uvicorn
import socket
import httpx

from vector_db import reset_faiss_index
from metadata_db import clear_catalog
from populate_db import populate
from main import app

def check_mongodb():
    try:
        with socket.create_connection(("localhost", 27017), timeout=2):
            print("✅ MongoDB is running on localhost:27017")
    except OSError:
        raise RuntimeError("❌ MongoDB is not running on localhost:27017")

async def check_triton():
    url = "http://localhost:8000/v2/health/ready"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=3)
            # print(response.status_code)
            if response.status_code == 200:
                print("✅ Triton server is ready at localhost:8000")
            else:
                raise RuntimeError("❌ Triton server responded but is not ready.")
    except Exception as e:
        raise RuntimeError(f"❌ Triton server is not running or not healthy: {e}")

async def main():
    # Set environment variables
    os.environ["MONGO_URI"] = "mongodb://localhost:27017"
    os.environ["TRITON_URL"] = "http://localhost:8000"

    # Service checks
    check_mongodb()
    await check_triton()

    # Reset
    reset_faiss_index()
    await clear_catalog()

    # Populate DB
    print("Populating product embeddings into FAISS and MongoDB...")
    await populate()
    print("✅ Population done.")

    # Start FastAPI app
    print("Starting FastAPI server at http://localhost:8003 ...")
    config = uvicorn.Config(app=app, host="0.0.0.0", port=8003, reload=True)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())