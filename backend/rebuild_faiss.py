import os
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

# Đường dẫn file dữ liệu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "image_embeddings.npy")

# Load mô hình CLIP
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Đọc bảng metadata (danh sách ảnh mẫu)
df = pd.read_csv(METADATA_PATH)
embeddings = []

# Duyệt từng ảnh, encode thành vector 768 chiều bằng CLIP
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
            # Mở ảnh → đưa qua CLIP → lấy vector đặc trưng 768 chiều
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            image_features = model.visual_projection(vision_outputs.pooler_output)
            # Chuẩn hóa L2 để dùng cosine similarity
            image_features = image_features / torch.norm(image_features, p=2, dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy()[0])
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

embeddings = np.array(embeddings).astype('float32')

print(f"Extracted shape: {embeddings.shape}")
# Lưu embeddings ra file backup
np.save(EMBEDDINGS_PATH, embeddings)

# Cập nhật metadata chỉ giữ các ảnh hợp lệ
new_df = df.iloc[valid_indices].reset_index(drop=True)
new_df.to_csv(METADATA_PATH, index=False)

# Tạo FAISS index — Inner Product trên vector đã normalize = Cosine Similarity
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Lưu FAISS index ra file (~15MB)
faiss.write_index(index, FAISS_INDEX_PATH)
print("Saved newly created FAISS generic index with Dimension", dim)
