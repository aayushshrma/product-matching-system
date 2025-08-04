# Product Matching System

This is a multimodal product matching system that uses image and text embeddings to identify the closest matching product from a catalog.

## Features

- Upload product image and description through a web interface
- Embeds input image using CLIP-based vision model via NVIDIA Triton Inference Server
- Embeds input text using CLIP tokenizer + text model via Triton
- Combines both embeddings and performs nearest neighbor search with FAISS
- Retrieves matching metadata from MongoDB
- Logs each match query with image and results
- Simple HTML UI for `/` and `/logs` routes

## Tech Stack

- **FastAPI** for backend server
- **MongoDB** for catalog and query logs
- **NVIDIA Triton Inference Server** to host quantized ONNX models
- **FAISS** for vector similarity search
- **Docker Compose** for service orchestration

## Setup

### Prerequisites

- Docker + NVIDIA Container Toolkit
- Python 3.12 (for running local utilities)
- Triton image: `nvcr.io/nvidia/tritonserver:25.05-py3`
- MongoDB image: `mongo:8.0.12`

### Run the System

```bash
docker compose up --build
```

Then open `http://localhost:8003` in your browser to use the matcher.

### Project Structure

```
app/
├── main.py                 # FastAPI routes & HTML interface
├── inference.py            # Triton inference functions
├── vector_db.py            # FAISS index management
├── metadata_db.py          # MongoDB interface for catalog
├── logs_db.py              # MongoDB interface for query logs
├── populate_db.py          # Adds image/text embeddings to FAISS + MongoDB
├── run.py          # main run script
model_repo/
├── vision_model/
│   └── config.pbtxt
├── text_model/
│   └── config.pbtxt
catalog/
├── boots.jpg               # Sample images
├── hoodie.jpg
├── tablet.jpg
├── table.jpg
```

## Useful Endpoints

- `/` – Upload product image + text via form
- `/match` – POST handler for matching product
- `/logs` – View recent matching logs in table

---