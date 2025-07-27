import faiss
import numpy as np

index = faiss.IndexFlatL2(512)
product_ids = []


async def add_embedding_to_index(product_id, embedding):
    index.add(np.array([embedding]).astype('float32'))
    product_ids.append(product_id)


async def search_nearest_embeddings(query_embedding, top_k=5):
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    result = [{"product_id": product_ids[i], "distance": float(D[0][j])} for j,i in enumerate(I[0])]
    return result