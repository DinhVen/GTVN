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

# --- Directional mirror pairs (label_left, label_right) ---
# When both appear in candidates, use flip-comparison to pick the correct direction
MIRROR_PAIRS = {
    # Cấm rẽ
    "P.123a": "P.123b",  # Cấm rẽ trái / Cấm rẽ phải
    "P.123b": "P.123a",
    # Cấm ô tô rẽ
    "P.103b": "P.103c",  # Cấm ô tô rẽ phải / Cấm ô tô rẽ trái
    "P.103c": "P.103b",
    # Cấm quay đầu
    "P.124a1": "P.124a2",
    "P.124a2": "P.124a1",
    "P.124b1": "P.124b2",
    "P.124b2": "P.124b1",
    # Cấm rẽ + quay đầu
    "P.124c": "P.124d",  # Cấm rẽ trái+quay đầu / Cấm rẽ phải+quay đầu
    "P.124d": "P.124c",
    "P.124e": "P.124f",
    "P.124f": "P.124e",
    # Biển nguy hiểm cua
    "W.201a": "W.201b",  # Cua trái / Cua phải
    "W.201b": "W.201a",
    "W.202a": "W.202b",
    "W.202b": "W.202a",
    # Biển hiệu lệnh rẽ
    "R.301a": "R.301b",
    "R.301b": "R.301a",
    "R.301e": "R.301f",
    "R.301f": "R.301e",
}

app = FastAPI(title="Traffic Sign Recognition API")

# --- CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

app.mount("/dataset_aug", StaticFiles(directory=os.path.join(BASE_DIR, "dataset_aug")), name="dataset_aug")

# --- Global Variables for Models and Data ---
model = None
processor = None
faiss_index = None
metadata_df = None
ood_text_feats = None  # Pre-computed OOD text embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_resources():
    global model, processor, faiss_index, metadata_df, ood_text_feats
    
    print("Loading CLIP model...")
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()  # Disable dropout for faster inference
    
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    if not os.path.exists(FAISS_INDEX_PATH):
        raise RuntimeError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    
    print(f"Loading Metadata from {METADATA_PATH}...")
    if not os.path.exists(METADATA_PATH):
        raise RuntimeError(f"Metadata file not found at {METADATA_PATH}")
    metadata_df = pd.read_csv(METADATA_PATH)
    
    # Pre-compute OOD text embeddings (these never change, no need to recompute per request)
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
    
    print("All resources loaded successfully!")

@app.get("/")
def read_root():
    return {"message": "Traffic Sign Recognition API is running!"}

@app.post("/search")
async def search_sign(file: UploadFile = File(...), top_k: int = 3):
    if file.content_type is None or not file.content_type.startswith("image/"):
        # We can try to proceed anyway if content_type is missing but we'll print a warning
        pass
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read image: {e}")

    # Extract features using CLIP
    try:
        # === STEP 1: Image Encoding (original + flipped for direction detection) ===
        image_flipped = ImageOps.mirror(image)
        image_inputs = processor(images=[image, image_flipped], return_tensors="pt").to(device)
        
        with torch.inference_mode():
            vision_outputs = model.vision_model(pixel_values=image_inputs["pixel_values"])
            both_features = model.visual_projection(vision_outputs.pooler_output)
            both_features = both_features / torch.norm(both_features, p=2, dim=-1, keepdim=True)
            
        query_vector = both_features[0:1].cpu().numpy()         # Original image
        flipped_vector = both_features[1:2].cpu().numpy()       # Flipped image
        img_feat_tensor = torch.tensor(query_vector).to(device)
        flipped_feat_tensor = torch.tensor(flipped_vector).to(device)
        
        # === STEP 2: OOD Check (using pre-computed OOD embeddings — zero cost) ===
        ood_sims = torch.matmul(img_feat_tensor, ood_text_feats.T)[0].cpu().numpy()
        print(f"OOD Sims: Traffic Sign = {ood_sims[0]:.4f}, Random = {ood_sims[1]:.4f}")
        
        if ood_sims[1] > ood_sims[0]:
            raise HTTPException(status_code=400, detail="Hình ảnh không hợp lệ! Vui lòng tải lên đúng đoạn cắt có chứa biển báo. (Hệ thống phát hiện đây không phải là biển báo)")
        
        # === STEP 3: FAISS Visual Search ===
        TOP_CANDIDATES = 30
        distances, indices = faiss_index.search(query_vector, max(top_k, TOP_CANDIDATES))
        
        # === STEP 4: Build re-ranking text prompts ===
        candidates = []
        rerank_prompts = []
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            row = metadata_df.iloc[idx].fillna("").to_dict()
            label = str(row.get("label", "Unknown"))
            meaning = str(row.get("meaning", ""))
            
            candidates.append({
                "idx": idx,
                "visual_score": float(distances[0][i]),
                "row": row,
                "label": label
            })
            rerank_prompts.append(f"A Vietnamese traffic sign that means: {meaning}")
        
        # === STEP 5: Text Re-ranking (single batch) ===
        if rerank_prompts:
            text_inputs = processor(text=rerank_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.inference_mode():
                text_outputs = model.text_model(**text_inputs)
                text_features = model.text_projection(text_outputs.pooler_output)
                text_features = text_features / torch.norm(text_features, p=2, dim=-1, keepdim=True)
                
                text_sims = torch.matmul(img_feat_tensor, text_features.T)[0].cpu().numpy()
            
            for idx_c, cand in enumerate(candidates):
                t_score = float(text_sims[idx_c])
                cand["text_score"] = t_score
                cand["final_score"] = cand["visual_score"] + (t_score * 1.8)

        # === STEP 6: Flip-Aware Direction Fix ===
        # For mirror pairs (left/right), compare original vs flipped image to determine direction
        cand_labels = {c["label"] for c in candidates}
        
        for cand in candidates:
            label = cand["label"]
            mirror = MIRROR_PAIRS.get(label)
            if mirror and mirror in cand_labels:
                # Both directions are in candidates — use flip to decide
                # Get the FAISS embedding of this candidate's reference image
                ref_embedding = faiss_index.reconstruct(int(cand["idx"]))
                ref_tensor = torch.tensor(ref_embedding).unsqueeze(0).to(device)
                
                orig_sim = torch.matmul(img_feat_tensor, ref_tensor.T)[0].item()
                flip_sim = torch.matmul(flipped_feat_tensor, ref_tensor.T)[0].item()
                
                # If flipped image matches this candidate BETTER than original,
                # then the actual sign is the MIRROR variant → penalize this one
                if flip_sim > orig_sim:
                    cand["final_score"] -= 0.30  # Strong penalize wrong direction
                    print(f"  FLIP-FIX: {label} penalized (orig={orig_sim:.4f}, flip={flip_sim:.4f})")
                else:
                    cand["final_score"] += 0.10  # Boost confirmed direction
                    print(f"  FLIP-FIX: {label} confirmed (orig={orig_sim:.4f}, flip={flip_sim:.4f})")

        # Sort by Multi-modal score
        candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
        
        # Deduplicate to show unique signs (fixes showing the same sign 3 times)
        seen_labels = set()
        results = []
        
        for cand in candidates:
            if cand["label"] in seen_labels: continue
            seen_labels.add(cand["label"])
            
            row = cand["row"]
            # Normalize score to 0-1 range: final = visual(0-1) + text(0-1)*1.8, max ~2.8
            display_score = min(cand["final_score"] / 2.8, 1.0)
            results.append({
                "rank": len(results) + 1,
                "score": display_score,  # Normalized so % is 0-100 and matches ranking order
                "label": cand["label"],
                "group": str(row.get("group", "Unknown")),
                "meaning": str(row.get("meaning", "No meaning available")),
                "advice": str(row.get("advice", "Tuân thủ luật giao thông")),
                "image_path": str(row.get("image_path", ""))
            })
            
            if len(results) >= top_k:
                break
                
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
