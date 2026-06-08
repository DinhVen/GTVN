import io
import os
import time
import traceback
from collections import Counter

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

# Cấu hình
MODEL_NAME = "openai/clip-vit-large-patch14"
IMAGE_SIZE = (224, 224)
TOP_CANDIDATES = 50
MAX_TOP_K = 10
SCORE_SCALE = 1.35

# Ngưỡng lọc ảnh không hợp lệ
OOD_SCORE_THRESHOLD = 0.70

# Điểm nhóm
GROUP_VOTING_TOP_N = 10
GROUP_MAJORITY_BONUS = 0.20
GROUP_MINORITY_PENALTY = 0.15

# Điểm hướng trái/phải
MIRROR_PENALTY = 0.30
MIRROR_BONUS = 0.10

# Các cặp biển trái/phải
MIRROR_PAIRS = {
    "P.123a": "P.123b", "P.123b": "P.123a",
    "P.103b": "P.103c", "P.103c": "P.103b",
    "P.124a1": "P.124a2", "P.124a2": "P.124a1",
    "P.124b1": "P.124b2", "P.124b2": "P.124b1",
    "P.124c": "P.124d", "P.124d": "P.124c",
    "P.124e": "P.124f", "P.124f": "P.124e",
    "W.201a": "W.201b", "W.201b": "W.201a",
    "W.202a": "W.202b", "W.202b": "W.202a",
    "R.301a": "R.301b", "R.301b": "R.301a",
    "R.301e": "R.301f", "R.301f": "R.301e",
}


# Đường dẫn và biến toàn cục
torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_aug")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"

model: CLIPModel | None = None
processor: CLIPProcessor | None = None
faiss_index: faiss.IndexFlatIP | None = None
metadata_records: list[dict] | None = None

# Hàm tiện ích
def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"{label} not found at {path}")


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(p=2, dim=-1, keepdim=True)


def _display_image_path(image_path: str) -> str:
    if not image_path:
        return ""
    folder = os.path.dirname(image_path).replace("\\", "/")
    candidate = f"{folder}/1_original.png"
    if os.path.exists(os.path.join(BASE_DIR, candidate)):
        return candidate
    return image_path.replace("\\", "/")


# CLIP encode
def _encode_images(images: list[Image.Image]) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        feats = model.visual_projection(
            model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        )
        return _normalize(feats)


# Pipeline nhận diện
def _search_faiss(query_vec: np.ndarray) -> list[dict]:
    distances, indices = faiss_index.search(query_vec, TOP_CANDIDATES)
    candidates = []
    for pos, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(metadata_records):
            continue
        row = metadata_records[idx]
        score = float(distances[0][pos])
        candidates.append({
            "idx": idx,
            "visual_score": score,
            "final_score": score,
            "row": row,
            "label": str(row.get("label", "Unknown")),
        })
    return candidates


def _check_ood(top1_score: float) -> None:
    print(f"  OOD: top1={top1_score:.4f}, threshold={OOD_SCORE_THRESHOLD}")
    if top1_score < OOD_SCORE_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail="Hình ảnh không hợp lệ! Vui lòng tải lên đúng đoạn cắt có chứa biển báo.",
        )


def _rerank_group_voting(candidates: list[dict]) -> None:
    if not candidates:
        return
    top_n = candidates[:GROUP_VOTING_TOP_N]
    group_counts = Counter(c["row"].get("group", "") for c in top_n)
    majority_group = group_counts.most_common(1)[0][0]
    print(f"  GROUP: majority={majority_group} (top {len(top_n)}: {dict(group_counts)})")
    for c in candidates:
        if c["row"].get("group", "") == majority_group:
            group_score = GROUP_MAJORITY_BONUS
        else:
            group_score = -GROUP_MINORITY_PENALTY
        c["final_score"] += group_score


def _fix_mirror(candidates: list[dict], query: torch.Tensor, flipped: torch.Tensor) -> None:
    labels_in_pool = {c["label"] for c in candidates}
    for c in candidates:
        mirror = MIRROR_PAIRS.get(c["label"])
        if not mirror or mirror not in labels_in_pool:
            continue
        ref = torch.from_numpy(faiss_index.reconstruct(int(c["idx"]))).unsqueeze(0).to(device)
        orig_sim = (query @ ref.T)[0].item()
        flip_sim = (flipped @ ref.T)[0].item()
        if flip_sim > orig_sim:
            mirror_score = -MIRROR_PENALTY
            c["final_score"] += mirror_score
            print(f"  MIRROR: {c['label']} penalty (orig={orig_sim:.4f}, flip={flip_sim:.4f})")
        else:
            mirror_score = MIRROR_BONUS
            c["final_score"] += mirror_score


def _format_results(candidates: list[dict], top_k: int) -> list[dict]:
    results, seen = [], set()
    for c in sorted(candidates, key=lambda x: x["final_score"], reverse=True):
        label = c["label"]
        if label in seen:
            continue
        seen.add(label)
        row = c["row"]
        results.append({
            "rank": len(results) + 1,
            "score": min(c["final_score"] / SCORE_SCALE, 0.99),
            "label": label,
            "group": str(row.get("group", "Unknown")),
            "meaning": str(row.get("meaning", "No meaning available")),
            "advice": str(row.get("advice", "Tuân thủ luật giao thông")),
            "image_path": _display_image_path(str(row.get("image_path", ""))),
        })
        if len(results) >= top_k:
            break
    return results


# FastAPI
app = FastAPI(title="Traffic Sign Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/dataset_aug", StaticFiles(directory=DATASET_DIR), name="dataset_aug")


@app.on_event("startup")
def load_resources():
    global model, processor, faiss_index, metadata_records

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    _require_file(FAISS_INDEX_PATH, "FAISS index")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    print(f"Loading metadata from {METADATA_PATH}...")
    _require_file(METADATA_PATH, "Metadata file")
    df = pd.read_csv(METADATA_PATH)
    metadata_records = df.fillna("").to_dict("records")

    if faiss_index.ntotal != len(metadata_records):
        print(
            f"WARNING: mismatch {faiss_index.ntotal} vectors vs {len(metadata_records)} records"
        )

    print(f"Ready! {faiss_index.ntotal} vectors, {len(metadata_records)} records.")


@app.get("/")
def health_check():
    return {"message": "Traffic Sign Recognition API is running!"}

#API
@app.post("/search")
async def search_sign(file: UploadFile = File(...), top_k: int = 3):
    if not all(r is not None for r in (model, processor, faiss_index, metadata_records)):
        raise HTTPException(status_code=503, detail="Server đang tải, vui lòng đợi.")

    # Kiểm tra và đọc ảnh
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Tệp tải lên phải là hình ảnh (JPG, PNG, JPEG).")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh: {exc}") from exc

    top_k = max(1, min(int(top_k), MAX_TOP_K))

    try:
        t0 = time.perf_counter()

        # Resize ảnh
        img = image.resize(IMAGE_SIZE, Image.BILINEAR)

        # Encode ảnh bằng CLIP
        query_feat = _encode_images([img])
        query_vec = query_feat[0:1].cpu().numpy()
        query_t = query_feat[0:1]

        # Tìm ảnh gần nhất và kiểm tra ảnh hợp lệ
        candidates = _search_faiss(query_vec)
        if candidates:
            _check_ood(candidates[0]["visual_score"])

        # Xếp hạng lại theo nhóm
        _rerank_group_voting(candidates)

        # Sửa nhầm trái/phải nếu cần
        labels = {c["label"] for c in candidates}
        if any(MIRROR_PAIRS.get(lb) in labels for lb in labels):
            flipped_feat = _encode_images([ImageOps.mirror(img)])
            _fix_mirror(candidates, query_t, flipped_feat[0:1])

        results = _format_results(candidates, top_k)

        print(f"  >> Inference: {(time.perf_counter() - t0) * 1000:.0f}ms")
        return {"results": results}

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
