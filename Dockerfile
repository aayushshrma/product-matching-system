FROM python:3.10-slim

WORKDIR /app

COPY ./app /app

RUN apt-get update && \
    apt-get install -y libglib2.0-0 && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pymongo \
        motor \
        requests \
        faiss-cpu \
        aiohttp \

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]