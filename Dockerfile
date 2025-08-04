FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app .

CMD ["python", "run.py"]