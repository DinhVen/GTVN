import io
import os
import time
import traceback

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

# Cấu hình //
MODEL_NAME = "openai/clip-vit-large-patch14"
IMAGE_SIZE = (224, 224)
TEXT_BATCH_SIZE = 64
TOP_CANDIDATES = 50
MAX_TOP_K = 10
TEXT_SCORE_WEIGHT = 1.8
DISPLAY_SCORE_DENOMINATOR = 1.6
MIRROR_WRONG_DIRECTION_PENALTY = 0.30
MIRROR_CORRECT_DIRECTION_BONUS = 0.10
DEFAULT_GROUP_DESC = "a Vietnamese traffic"
RESAMPLE_FILTER = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

# Hai prompt dùng cho OOD check (bước 3).
OOD_PROMPTS = [
    "A close-up photo of a traffic sign",
    "A photo of a signature, text document, animal, scenery, or random object",
]

# Ánh xạ nhóm biển → mô tả hình dạng + màu sắc (bước 5).
GROUP_DESC = {
    "cấm": "a red circular prohibitory",
    "biển cấm": "a red circular prohibitory",
    "nguy hiểm": "a yellow triangular warning",
    "hiệu lệnh": "a blue circular mandatory",
    "chỉ dẫn": "a blue rectangular information",
    "biển phụ": "a supplementary",
}

# 11 cặp biển đối xứng trái/phải (bước 6).
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

# Gộp label hiển thị (biển cùng hình khác số).
LABEL_ALIASES = {
    "P.127_50": "P.127",
}


# Khởi tạo đường dẫn và biến toàn cục //
torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_aug")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tài nguyên được load khi startup, dùng chung cho tất cả request.
model: CLIPModel | None = None
processor: CLIPProcessor | None = None
faiss_index: faiss.IndexFlatIP | None = None
metadata_records: list[dict] | None = None
ood_text_feats: torch.Tensor | None = None
all_text_feats: torch.Tensor | None = None


# Hàm tiện ích//
def _require_file(path: str, label: str) -> None:
    """Kiểm tra file tồn tại, raise lỗi nếu không."""
    if not os.path.exists(path):
        raise RuntimeError(f"{label} not found at {path}")


def _normalize(features: torch.Tensor) -> torch.Tensor:
    """Chuẩn hóa L2 — đưa vector về độ dài 1."""
    return features / features.norm(p=2, dim=-1, keepdim=True)


def _display_label(label: str) -> str:
    """Trả label hiển thị (gộp alias nếu có)."""
    return LABEL_ALIASES.get(label, label)


def _display_image_path(image_path: str) -> str:
    """Luôn trả ảnh gốc 1_original.png thay vì ảnh augment."""
    if not image_path:
        return ""
    folder = os.path.dirname(image_path).replace("\\", "/")
    candidate = f"{folder}/1_original.png"
    if os.path.exists(os.path.join(BASE_DIR, candidate)):
        return candidate
    return image_path.replace("\\", "/")


def _normalize_top_k(top_k: int) -> int:
    """Giới hạn số kết quả trả về."""
    return max(1, min(int(top_k), MAX_TOP_K))


def _resources_ready() -> bool:
    """Kiểm tra tài nguyên AI đã load xong chưa."""
    resources = (model, processor, faiss_index, metadata_records, ood_text_feats, all_text_feats)
    return all(resource is not None for resource in resources)


def _group_description(group: str) -> str:
    """Lấy mô tả hình dạng/màu sắc theo nhóm biển."""
    normalized_group = str(group or "").lower().strip()
    return GROUP_DESC.get(normalized_group, DEFAULT_GROUP_DESC)


def _build_text_prompt(record: dict) -> str:
    """Tạo prompt text từ group và meaning trong metadata."""
    group_desc = _group_description(record.get("group", ""))
    meaning = record.get("meaning", "")
    return f"A photo of {group_desc} sign that means: {meaning}"


def _has_mirror_pair(candidates: list[dict]) -> bool:
    """Kiểm tra pool ứng viên có cặp trái/phải không."""
    labels = {c["label"] for c in candidates}
    return any(MIRROR_PAIRS.get(label) in labels for label in labels)


async def _read_upload_image(file: UploadFile) -> Image.Image:
    """Đọc file upload và chuyển sang ảnh RGB."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Tệp tải lên phải là hình ảnh (JPG, PNG, JPEG).")
    try:
        return Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh: {exc}") from exc


# Encode (mã hóa) bằng CLIP //
def _encode_texts(prompts: list[str]) -> torch.Tensor:
    """Mã hóa danh sách câu text → tensor [N, 768]."""
    batches = []
    with torch.inference_mode():
        for i in range(0, len(prompts), TEXT_BATCH_SIZE):
            inputs = processor(
                text=prompts[i : i + TEXT_BATCH_SIZE],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            feats = model.text_projection(model.text_model(**inputs).pooler_output)
            batches.append(_normalize(feats).cpu())
    return torch.cat(batches).to(device)


def _encode_images(images: list[Image.Image]) -> torch.Tensor:
    """Mã hóa danh sách ảnh PIL → tensor [N, 768]."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        feats = model.visual_projection(
            model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        )
        return _normalize(feats)


# Pipeline 6 bước //
def _check_ood(query: torch.Tensor) -> None:
    """Bước 3 — OOD check: reject nếu ảnh không phải biển báo."""
    sims = (query @ ood_text_feats.T)[0].cpu().numpy()
    print(f"OOD: sign={sims[0]:.4f}, random={sims[1]:.4f}")
    if sims[1] > sims[0]:
        raise HTTPException(
            status_code=400,
            detail="Hình ảnh không hợp lệ! Vui lòng tải lên đúng đoạn cắt có chứa biển báo.",
        )


def _search_faiss(query_vec: np.ndarray) -> tuple[list[dict], list[int]]:
    """Bước 4 — FAISS search: tìm top-K ứng viên gần nhất."""
    distances, indices = faiss_index.search(query_vec, TOP_CANDIDATES)
    candidates, cand_indices = [], []
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
        cand_indices.append(idx)
    return candidates, cand_indices


def _rerank_text(candidates: list[dict], cand_indices: list[int], query: torch.Tensor) -> None:
    """Bước 5 — Text re-ranking: cộng điểm text vào final_score."""
    if not cand_indices:
        return
    text_sims = (query @ all_text_feats[cand_indices].T)[0].cpu().numpy()
    for i, c in enumerate(candidates):
        c["final_score"] = c["visual_score"] + float(text_sims[i]) * TEXT_SCORE_WEIGHT


def _fix_mirror(candidates: list[dict], query: torch.Tensor, flipped: torch.Tensor) -> None:
    """Bước 6 — Mirror fix: sửa nhầm trái/phải cho cặp biển đối xứng."""
    labels_in_pool = {c["label"] for c in candidates}
    for c in candidates:
        mirror = MIRROR_PAIRS.get(c["label"])
        if not mirror or mirror not in labels_in_pool:
            continue
        ref = torch.from_numpy(faiss_index.reconstruct(int(c["idx"]))).unsqueeze(0).to(device)
        orig_sim = (query @ ref.T)[0].item()
        flip_sim = (flipped @ ref.T)[0].item()
        if flip_sim > orig_sim:
            c["final_score"] -= MIRROR_WRONG_DIRECTION_PENALTY
            print(f"  MIRROR: {c['label']} penalty (orig={orig_sim:.4f}, flip={flip_sim:.4f})")
        else:
            c["final_score"] += MIRROR_CORRECT_DIRECTION_BONUS


def _format_results(candidates: list[dict], top_k: int) -> list[dict]:
    """Lọc trùng theo label, tính % hiển thị, trả top-K."""
    results, seen = [], set()
    for c in sorted(candidates, key=lambda x: x["final_score"], reverse=True):
        label = _display_label(c["label"])
        if label in seen:
            continue
        seen.add(label)
        row = c["row"]
        results.append({
            "rank": len(results) + 1,
            "score": min(c["final_score"] / DISPLAY_SCORE_DENOMINATOR, 1.0),
            "label": label,
            "group": str(row.get("group", "Unknown")),
            "meaning": str(row.get("meaning", "No meaning available")),
            "advice": str(row.get("advice", "Tuân thủ luật giao thông")),
            "image_path": _display_image_path(str(row.get("image_path", ""))),
        })
        if len(results) >= top_k:
            break
    return results


# FastAPI //
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
    """Load model, FAISS index, metadata và pre-compute text embeddings."""
    global model, processor, faiss_index, metadata_records, ood_text_feats, all_text_feats

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
            "WARNING: FAISS index and metadata length mismatch "
            f"({faiss_index.ntotal} vectors vs {len(metadata_records)} records)."
        )

    print("Pre-computing OOD text embeddings...")
    ood_text_feats = _encode_texts(OOD_PROMPTS)

    print("Pre-computing text embeddings for all signs...")
    all_text_feats = _encode_texts([_build_text_prompt(record) for record in metadata_records])

    print(f"Ready! {faiss_index.ntotal} vectors, {len(metadata_records)} records.")


@app.get("/")
def health_check():
    return {"message": "Traffic Sign Recognition API is running!"}


@app.post("/search")
async def search_sign(file: UploadFile = File(...), top_k: int = 3):
    """API nhận diện biển báo — pipeline 6 bước."""
    if not _resources_ready():
        raise HTTPException(status_code=503, detail="Server đang tải, vui lòng đợi.")

    image = await _read_upload_image(file)
    top_k = _normalize_top_k(top_k)

    try:
        t0 = time.perf_counter()

        # Bước 1: Resize
        img = image.resize(IMAGE_SIZE, RESAMPLE_FILTER)

        # Bước 2: CLIP encode
        query_feat = _encode_images([img])
        query_vec = query_feat[0:1].cpu().numpy()
        query_t = query_feat[0:1]

        # Bước 3: OOD check
        _check_ood(query_t)

        # Bước 4: FAISS search
        candidates, cand_idx = _search_faiss(query_vec)

        # Bước 5: Text re-ranking
        _rerank_text(candidates, cand_idx, query_t)

        # Bước 6: Mirror fix (chỉ khi có cặp đối xứng trong ứng viên)
        if _has_mirror_pair(candidates):
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
