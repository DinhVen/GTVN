"""
Xây dựng FAISS index từ toàn bộ ảnh trong metadata.csv.

Quy trình:
  1. Load CLIP model
  2. Duyệt metadata.csv → encode từng ảnh thành vector 768 chiều
  3. Chuẩn hóa L2 (normalize)
  4. Tạo FAISS IndexFlatIP (Inner Product)
  5. Lưu faiss_index.faiss + image_embeddings.npy

Chạy: python backend/rebuild_faiss.py
"""

import os

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-large-patch14"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "image_embeddings.npy")


def _normalize(features: torch.Tensor) -> torch.Tensor:
    """Chuẩn hóa L2 — đưa vector về độ dài 1."""
    return features / features.norm(p=2, dim=-1, keepdim=True)


def _encode_image(path: str, model, processor, device: str) -> np.ndarray:
    """Encode 1 ảnh → vector 768 chiều (numpy)."""
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        feats = model.visual_projection(
            model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        )
        return _normalize(feats).cpu().numpy()[0]


def main() -> None:
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    df = pd.read_csv(METADATA_PATH)
    print(f"Metadata: {len(df)} ảnh")

    # Encode tất cả ảnh
    embeddings, valid_indices = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
        path = os.path.join(BASE_DIR, str(row["image_path"]))
        if not os.path.exists(path):
            print(f"  Skip (not found): {path}")
            continue
        try:
            embeddings.append(_encode_image(path, model, processor, device))
            valid_indices.append(idx)
        except Exception as exc:
            print(f"  Error: {path} — {exc}")

    if not embeddings:
        raise RuntimeError("Không có ảnh nào được encode thành công!")

    embeddings = np.array(embeddings, dtype="float32")
    print(f"Embeddings: {embeddings.shape}")

    # Lưu embeddings backup
    np.save(EMBEDDINGS_PATH, embeddings)

    # Lọc metadata chỉ giữ ảnh hợp lệ
    df = df.iloc[valid_indices].reset_index(drop=True)
    df.to_csv(METADATA_PATH, index=False)

    # Tạo FAISS index (IndexFlatIP — Inner Product, tương đương cosine khi đã normalize)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved: {FAISS_INDEX_PATH} ({index.ntotal} vectors, dim={embeddings.shape[1]})")


if __name__ == "__main__":
    main()
