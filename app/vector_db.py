import faiss
import numpy as np

index = faiss.IndexFlatL2(1024)  # argmin||x - xi||
product_ids = []

async def add_embedding_to_index(product_id, image_emb, text_emb):
    combined_emb = np.concatenate([image_emb, text_emb], axis=-1)
    index.add(np.array([combined_emb]).astype('float32'))
    product_ids.append(product_id)

async def search_nearest_embeddings(image_emb, text_emb, top_k=5):
    combined_emb = np.concatenate([image_emb, text_emb], axis=-1)
    dist, idx = index.search(np.array([combined_emb]).astype('float32'), top_k)
    result = [{"product_id": product_ids[i], "distance": float(dist[0][j])} for j,i in enumerate(idx[0])]
    return result

def reset_faiss_index():
    global index, product_ids
    index.reset()
    product_ids = []