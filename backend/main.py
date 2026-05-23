import os
import io
import numpy as np
import pandas as pd
import faiss
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageOps
import time

# Tận dụng hết CPU cores để PyTorch xử lý nhanh hơn
torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(1)

# Bảng cặp biển trái/phải — CLIP không phân biệt được hướng nên liệt kê ra để xử lý riêng
MIRROR_PAIRS = {
    "P.123a": "P.123b",   # Cấm rẽ trái / Cấm rẽ phải
    "P.123b": "P.123a",
    "P.103b": "P.103c",   # Cấm ô tô rẽ phải / Cấm ô tô rẽ trái
    "P.103c": "P.103b",
    "P.124a1": "P.124a2",  # Cấm quay đầu
    "P.124a2": "P.124a1",
    "P.124b1": "P.124b2",
    "P.124b2": "P.124b1",
    "P.124c": "P.124d",   # Cấm rẽ trái+quay đầu / Cấm rẽ phải+quay đầu
    "P.124d": "P.124c",
    "P.124e": "P.124f",
    "P.124f": "P.124e",
    "W.201a": "W.201b",   # Cua trái / Cua phải
    "W.201b": "W.201a",
    "W.202a": "W.202b",
    "W.202b": "W.202a",
    "R.301a": "R.301b",   # Hiệu lệnh rẽ trái / rẽ phải
    "R.301b": "R.301a",
    "R.301e": "R.301f",
    "R.301f": "R.301e",
}

# Tạo ứng dụng FastAPI
app = FastAPI(title="Traffic Sign Recognition API")

# Cho phép Frontend gọi API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn đến thư mục dữ liệu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

# Cho phép truy cập ảnh mẫu từ frontend
app.mount("/dataset_aug", StaticFiles(directory=os.path.join(BASE_DIR, "dataset_aug")), name="dataset_aug")

# Biến toàn cục — sẽ được gán giá trị khi server khởi động
model = None
processor = None
faiss_index = None
metadata_df = None
ood_text_feats = None   # Vector text OOD đã tính sẵn
all_text_feats = None   # Vector text của TẤT CẢ biển đã tính sẵn
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hàm khởi động — chạy 1 lần khi bật server, load mọi thứ vào RAM
@app.on_event("startup")
def load_resources():
    global model, processor, faiss_index, metadata_df, ood_text_feats, all_text_feats
    
    # 1. Load mô hình CLIP (428 triệu tham số)
    print("Loading CLIP model...")
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()  # Tắt dropout để chạy nhanh hơn
    
    # 2. Load FAISS index (4875 vectors đã tính sẵn)
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    if not os.path.exists(FAISS_INDEX_PATH):
        raise RuntimeError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    
    # 3. Load bảng metadata (tên biển, nhóm, ý nghĩa, lời khuyên)
    print(f"Loading Metadata from {METADATA_PATH}...")
    if not os.path.exists(METADATA_PATH):
        raise RuntimeError(f"Metadata file not found at {METADATA_PATH}")
    metadata_df = pd.read_csv(METADATA_PATH)
    
    # 4. Tính sẵn 2 câu text để kiểm tra ảnh có phải biển báo không (OOD check)
    print("Pre-computing OOD text embeddings...")
    ood_prompts = [
        "A close-up photo of a traffic sign",
        "A photo of a signature, text document, animal, scenery, or random object"
    ]
    ood_inputs = processor(text=ood_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        ood_outputs = model.text_model(**ood_inputs)
        ood_text_feats = model.text_projection(ood_outputs.pooler_output)
        ood_text_feats = ood_text_feats / torch.norm(ood_text_feats, p=2, dim=-1, keepdim=True)
    
    # 5. Tính sẵn text embeddings cho TẤT CẢ 4875 biển (tối ưu tốc độ quan trọng nhất!)
    #    Mỗi biển có 1 câu mô tả → CLIP text encode → vector 768 chiều
    #    Tính 1 lần ở đây, khi user gửi ảnh chỉ cần tra bảng (~1ms thay vì ~500ms)
    print("Pre-computing text embeddings for all signs...")
    all_prompts = [f"A Vietnamese traffic sign that means: {str(row.get('meaning', ''))}" 
                   for _, row in metadata_df.iterrows()]
    
    BATCH_SIZE = 64
    text_embeds_list = []
    with torch.inference_mode():
        for i in range(0, len(all_prompts), BATCH_SIZE):
            batch = all_prompts[i:i+BATCH_SIZE]
            text_inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
            text_outputs = model.text_model(**text_inputs)
            text_features = model.text_projection(text_outputs.pooler_output)
            text_features = text_features / torch.norm(text_features, p=2, dim=-1, keepdim=True)
            text_embeds_list.append(text_features.cpu())
    
    all_text_feats = torch.cat(text_embeds_list, dim=0).to(device)
    print(f"Pre-computed {all_text_feats.shape[0]} text embeddings!")
    
    print("All resources loaded successfully!")

# API kiểm tra server
@app.get("/")
def read_root():
    return {"message": "Traffic Sign Recognition API is running!"}

# === API CHÍNH: Nhận ảnh → trả kết quả nhận diện ===
@app.post("/search")
async def search_sign(file: UploadFile = File(...), top_k: int = 3):
    # Kiểm tra file upload có phải ảnh không
    if file.content_type is None or not file.content_type.startswith("image/"):
        pass
        
    # Đọc ảnh từ request
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read image: {e}")

    try:
        t0 = time.perf_counter()
        
        # BƯỚC 1: Encode ảnh thành vector 768 chiều bằng CLIP
        # Thu nhỏ về 224x224 (kích thước CLIP yêu cầu) cho nhanh
        image_small = image.resize((224, 224), Image.LANCZOS)
        # Lật ảnh để dùng ở bước 6 (phân biệt trái/phải)
        image_flipped = ImageOps.mirror(image_small)
        # Encode cả ảnh gốc và ảnh lật cùng lúc (1 lần chạy AI)
        image_inputs = processor(images=[image_small, image_flipped], return_tensors="pt").to(device)
        
        with torch.inference_mode():
            vision_outputs = model.vision_model(pixel_values=image_inputs["pixel_values"])
            both_features = model.visual_projection(vision_outputs.pooler_output)
            # Chuẩn hóa L2 để dùng cosine similarity
            both_features = both_features / torch.norm(both_features, p=2, dim=-1, keepdim=True)
            
        query_vector = both_features[0:1].cpu().numpy()         # Vector ảnh gốc
        flipped_vector = both_features[1:2].cpu().numpy()       # Vector ảnh lật
        img_feat_tensor = torch.tensor(query_vector).to(device)
        flipped_feat_tensor = torch.tensor(flipped_vector).to(device)
        
        # BƯỚC 2: Kiểm tra ảnh có phải biển báo không (OOD check)
        # So vector ảnh với 2 câu: "ảnh biển báo" vs "ảnh vật thể ngẫu nhiên"
        ood_sims = torch.matmul(img_feat_tensor, ood_text_feats.T)[0].cpu().numpy()
        print(f"OOD Sims: Traffic Sign = {ood_sims[0]:.4f}, Random = {ood_sims[1]:.4f}")
        
        # Nếu giống "vật thể ngẫu nhiên" hơn → từ chối
        if ood_sims[1] > ood_sims[0]:
            raise HTTPException(status_code=400, detail="Hình ảnh không hợp lệ! Vui lòng tải lên đúng đoạn cắt có chứa biển báo. (Hệ thống phát hiện đây không phải là biển báo)")
        
        # BƯỚC 3: Tìm kiếm FAISS — so vector ảnh với 4875 vectors mẫu, lấy top 30
        TOP_CANDIDATES = 30
        distances, indices = faiss_index.search(query_vector, max(top_k, TOP_CANDIDATES))
        
        # BƯỚC 4: Lấy thông tin 30 ứng viên từ metadata
        candidates = []
        candidate_indices = []
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            row = metadata_df.iloc[idx].fillna("").to_dict()
            label = str(row.get("label", "Unknown"))
            
            candidates.append({
                "idx": idx,
                "visual_score": float(distances[0][i]),
                "row": row,
                "label": label
            })
            candidate_indices.append(idx)
        
        # BƯỚC 5: Xếp hạng lại bằng text (tra bảng đã tính sẵn, không cần chạy AI)
        # So vector ảnh với mô tả text của từng ứng viên → ai giống nhất?
        if candidate_indices:
            cand_text_feats = all_text_feats[candidate_indices]
            text_sims = torch.matmul(img_feat_tensor, cand_text_feats.T)[0].cpu().numpy()
            
            for idx_c, cand in enumerate(candidates):
                t_score = float(text_sims[idx_c])
                cand["text_score"] = t_score
                # Điểm cuối = điểm hình + (điểm text × 1.8)
                cand["final_score"] = cand["visual_score"] + (t_score * 1.8)

        # BƯỚC 6: Sửa hướng trái/phải cho các cặp biển mirror
        # Nếu cả 2 hướng đều có trong kết quả → so ảnh gốc vs ảnh lật để chọn đúng
        cand_labels = {c["label"] for c in candidates}
        
        for cand in candidates:
            label = cand["label"]
            mirror = MIRROR_PAIRS.get(label)
            if mirror and mirror in cand_labels:
                # Lấy vector mẫu của biển này từ FAISS
                ref_embedding = faiss_index.reconstruct(int(cand["idx"]))
                ref_tensor = torch.tensor(ref_embedding).unsqueeze(0).to(device)
                
                # So: ảnh gốc giống mẫu bao nhiêu vs ảnh lật giống mẫu bao nhiêu
                orig_sim = torch.matmul(img_feat_tensor, ref_tensor.T)[0].item()
                flip_sim = torch.matmul(flipped_feat_tensor, ref_tensor.T)[0].item()
                
                # Ảnh lật giống hơn → biển thật là hướng ngược → phạt điểm
                if flip_sim > orig_sim:
                    cand["final_score"] -= 0.30  # Phạt nặng sai hướng
                    print(f"  FLIP-FIX: {label} penalized (orig={orig_sim:.4f}, flip={flip_sim:.4f})")
                else:
                    cand["final_score"] += 0.10  # Thưởng đúng hướng
                    print(f"  FLIP-FIX: {label} confirmed (orig={orig_sim:.4f}, flip={flip_sim:.4f})")

        # Sắp xếp theo điểm cao nhất
        candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
        
        # Lọc trùng — không hiện cùng 1 biển nhiều lần
        seen_labels = set()
        results = []
        
        for cand in candidates:
            if cand["label"] in seen_labels: continue
            seen_labels.add(cand["label"])
            
            row = cand["row"]
            # Chuẩn hóa điểm về 0-100% (max lý thuyết ~2.8)
            display_score = min(cand["final_score"] / 2.8, 1.0)
            results.append({
                "rank": len(results) + 1,
                "score": display_score,
                "label": cand["label"],
                "group": str(row.get("group", "Unknown")),
                "meaning": str(row.get("meaning", "No meaning available")),
                "advice": str(row.get("advice", "Tuân thủ luật giao thông")),
                "image_path": str(row.get("image_path", ""))
            })
            
            if len(results) >= top_k:
                break
        
        print(f"  >> Total inference: {(time.perf_counter()-t0)*1000:.0f}ms")
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
