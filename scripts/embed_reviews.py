import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = "data/processed"
VECTOR_DIR = "vectors"
os.makedirs(VECTOR_DIR, exist_ok=True)

# all-MiniLM-L6-v2: fast, small, good quality
# downloads ~90MB on first run
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_app(app_name):
    path = f"{PROCESSED_DIR}/{app_name}.parquet"
    df = pd.read_parquet(path)

    texts = df["review_text"].tolist()

    print(f"Embedding {app_name} ({len(texts)} reviews)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # build FAISS index
    dim = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after normalization
    index.add(embeddings)

    # save index
    faiss.write_index(index, f"{VECTOR_DIR}/{app_name}.index")

    # save embeddings as numpy array (needed for BERTopic later)
    np.save(f"{VECTOR_DIR}/{app_name}_embeddings.npy", embeddings)

    print(f"  {app_name}: {index.ntotal} vectors saved\n")

if __name__ == "__main__":
    apps = [
        f.replace(".parquet", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".parquet") and f != "all_apps.parquet"
    ]

    for app in sorted(apps):
        embed_app(app)

    print("All embeddings done.")