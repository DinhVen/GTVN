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


def normalize(features: torch.Tensor) -> torch.Tensor:
    return features / torch.norm(features, p=2, dim=-1, keepdim=True)


def encode_image(image_path: str, model: CLIPModel, processor: CLIPProcessor, device: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(outputs.pooler_output)
        return normalize(features).cpu().numpy()[0]


def build_embeddings(df: pd.DataFrame, model: CLIPModel, processor: CLIPProcessor, device: str):
    embeddings = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(BASE_DIR, str(row["image_path"]))
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue

        try:
            embeddings.append(encode_image(image_path, model, processor, device))
            valid_indices.append(idx)
        except Exception as exc:
            print(f"Error reading {image_path}: {exc}")

    return np.array(embeddings).astype("float32"), valid_indices


def save_faiss_index(embeddings: np.ndarray) -> None:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Saved newly created FAISS generic index with Dimension", embeddings.shape[1])


def main() -> None:
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    df = pd.read_csv(METADATA_PATH)

    print("Extracting features (This might take a few minutes)...")
    embeddings, valid_indices = build_embeddings(df, model, processor, device)
    if len(embeddings) == 0:
        raise RuntimeError("No valid image embeddings were generated")

    print(f"Extracted shape: {embeddings.shape}")
    np.save(EMBEDDINGS_PATH, embeddings)

    cleaned_df = df.iloc[valid_indices].reset_index(drop=True)
    cleaned_df.to_csv(METADATA_PATH, index=False)

    save_faiss_index(embeddings)


if __name__ == "__main__":
    main()
