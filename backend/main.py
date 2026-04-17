import os
import io
import pandas as pd
import faiss
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

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
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_resources():
    global model, processor, faiss_index, metadata_df
    
    print("Loading CLIP model...")
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    if not os.path.exists(FAISS_INDEX_PATH):
        raise RuntimeError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    
    print(f"Loading Metadata from {METADATA_PATH}...")
    if not os.path.exists(METADATA_PATH):
        raise RuntimeError(f"Metadata file not found at {METADATA_PATH}")
    metadata_df = pd.read_csv(METADATA_PATH)
    
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
        inputs = processor(images=image, return_tensors="pt").to(device)
        dummy_text = processor(text=["dummy"], return_tensors="pt").to(device)
        inputs.update(dummy_text)
        
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds
            
        image_features = image_features / torch.norm(image_features, p=2, dim=-1, keepdim=True)
        query_vector = image_features.cpu().numpy()
        
        # VLP Re-ranking: Retrieve more candidates and re-rank with Text
        TOP_CANDIDATES = 15
        distances, indices = faiss_index.search(query_vector, max(top_k, TOP_CANDIDATES))
        
        candidates = []
        text_list = []
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            row = metadata_df.iloc[idx].fillna("").to_dict()
            
            label = str(row.get("label", "Unknown"))
            meaning = str(row.get("meaning", ""))
            
            # Text prompt for CLIP text embeddings
            desc = f"Biển báo giao thông: {meaning}. Ký hiệu: {label}."
            
            candidates.append({
                "idx": idx,
                "visual_score": float(distances[0][i]),
                "row": row,
                "label": label
            })
            text_list.append(desc)
            
        if text_list:
            # Re-ranking using Text similarity
            text_inputs = processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                # Extract text features safely and FAST without triggering Vision branch
                text_outputs = model.text_model(**text_inputs)
                text_features = text_outputs.pooler_output
                text_features = model.text_projection(text_features)
                text_features = text_features / torch.norm(text_features, p=2, dim=-1, keepdim=True)
                
                # compute cosine similarity between original query image embed and text branch embeds
                img_feat_tensor = torch.tensor(query_vector).to(device)
                text_sims = torch.matmul(img_feat_tensor, text_features.T)[0].cpu().numpy()
                
            # Combine scores with weights (giving text score a solid impact)
            for idx_c, cand in enumerate(candidates):
                t_score = float(text_sims[idx_c])
                cand["text_score"] = t_score
                cand["final_score"] = cand["visual_score"] + (t_score * 0.85)

            # Sort by newly computed Multi-modal score
            candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
        
        # Deduplicate to show unique signs (fixes showing the same sign 3 times)
        seen_labels = set()
        results = []
        
        for cand in candidates:
            if cand["label"] in seen_labels: continue
            seen_labels.add(cand["label"])
            
            row = cand["row"]
            results.append({
                "rank": len(results) + 1,
                "score": cand["visual_score"], # Using original visual score for UI display transparency
                "label": cand["label"],
                "group": str(row.get("group", "Unknown")),
                "meaning": str(row.get("meaning", "No meaning available")),
                "advice": str(row.get("advice", "Tuân thủ luật giao thông")),
                "image_path": str(row.get("image_path", ""))
            })
            
            if len(results) >= top_k:
                break
                
        return {"results": results}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
