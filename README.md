# Product Matching System

This project implements an end-to-end image and text product matching system using FastAPI, Triton Inference Server, ONNX models, MongoDB, and Docker.

---

## 🧱 Project Structure

```
product-matching-system/
├── app/
│   ├── main.py                  # FastAPI application entry point
│   ├── inference.py             # Inference logic for image & text
│   ├── populate_db.py           # Populates the DB with embeddings
│   ├── logs_db.py               # Handles logging
│   └── metadata_db.py           # Handles product metadata
│
├── model_repo/
│   ├── text_model/
│   │   └── 1/
│   │       ├── model.onnx
│   │       └── config.pbtxt
│   └── vision_model/
│       └── 1/
│           ├── model.onnx
│           └── config.pbtxt
│
├── quantize_model.py           # Optional quantization script
├── text_model.plan             # Optional plan file (for TensorRT)
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 🧪 Requirements

- Docker
- Python 3.11+
- NVIDIA GPU (optional for GPU acceleration)
- MongoDB
- Triton Inference Server

---

## 🚀 Setup and Run

1. **Build and Start Services**

```bash
docker compose up --build
```

2. **Verify Containers**

```bash
docker ps
```

3. **Run Population Script**

```bash
docker exec -it fastapi_app python app/populate_db.py
```

This script loads embeddings for all product entries using your ONNX models and stores them in MongoDB.

---

## ⚙️ Environment Variables

Set inside `docker-compose.yml`:

```yaml
environment:
  - MONGO_URI=mongodb://mongodb:27017
  - TRITON_URL=http://triton:8000
```

---

## 🔌 Model Input/Output

### Image Model (`vision_model`)
- Input: `input_image` shape `[1, 3, 224, 224]`, dtype: FP32
- Output: `embedding` shape `[1, 512]`, dtype: FP32

### Text Model (`text_model`)
- Input: `input_ids`, `attention_mask`, shape `[77]`, dtype: INT32
- Output: `embedding`, shape `[1, 512]`, dtype: FP32

---

## ✅ Notes

- Use `.dockerignore` to exclude large or unnecessary local files.
- `.onnx` files are served by Triton.
- `config.pbtxt` must match model's actual input/output signatures.

---

## 🧠 Inference Logic

Implemented in `inference.py`. Communicates with Triton at:
```
http://triton:8000/v2/models/{model_name}/infer
```

- Text and image embeddings are returned as NumPy arrays.
- These embeddings are then stored in MongoDB or compared for similarity.

---

## 📫 API (Optional)

If you’re running FastAPI for inference, it will be served at:
```
http://localhost:8003/
```