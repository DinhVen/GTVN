import os
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "image_embeddings.npy")

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

df = pd.read_csv(METADATA_PATH)
embeddings = []

print("Extracting features (This might take a few minutes)...")
model.eval()

valid_indices = []

with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(BASE_DIR, str(row['image_path']))
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            image_features = model.visual_projection(vision_outputs.pooler_output)
            image_features = image_features / torch.norm(image_features, p=2, dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy()[0])
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

embeddings = np.array(embeddings).astype('float32')

print(f"Extracted shape: {embeddings.shape}")
np.save(EMBEDDINGS_PATH, embeddings)

# Filter dataset to only valid images (just in case some were missing)
new_df = df.iloc[valid_indices].reset_index(drop=True)
new_df.to_csv(METADATA_PATH, index=False)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim) # Cosine similarity = Inner Product of normalized vectors
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)
print("Saved newly created FAISS generic index with Dimension", dim)
