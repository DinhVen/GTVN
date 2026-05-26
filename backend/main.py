import io
import os
import time
import traceback

import faiss
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-large-patch14"
IMAGE_SIZE = (224, 224)
TEXT_BATCH_SIZE = 64
TOP_CANDIDATES = 30
TEXT_SCORE_WEIGHT = 1.8
DISPLAY_SCORE_DENOMINATOR = 2.8
MIRROR_WRONG_DIRECTION_PENALTY = 0.30
MIRROR_CORRECT_DIRECTION_BONUS = 0.10

OOD_PROMPTS = [
    "A close-up photo of a traffic sign",
    "A photo of a signature, text document, animal, scenery, or random object",
]

# Các cặp biển dễ nhầm trái/phải.
MIRROR_PAIRS = {
    "P.123a": "P.123b",
    "P.123b": "P.123a",
    "P.103b": "P.103c",
    "P.103c": "P.103b",
    "P.124a1": "P.124a2",
    "P.124a2": "P.124a1",
    "P.124b1": "P.124b2",
    "P.124b2": "P.124b1",
    "P.124c": "P.124d",
    "P.124d": "P.124c",
    "P.124e": "P.124f",
    "P.124f": "P.124e",
    "W.201a": "W.201b",
    "W.201b": "W.201a",
    "W.202a": "W.202b",
    "W.202b": "W.202a",
    "R.301a": "R.301b",
    "R.301b": "R.301a",
    "R.301e": "R.301f",
    "R.301f": "R.301e",
}

LABEL_ALIASES = {
    "P.127_50": "P.127",
}

torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_aug")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
processor = None
faiss_index = None
metadata_df = None
ood_text_feats = None
all_text_feats = None


def require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"{label} not found at {path}")


def normalize(features: torch.Tensor) -> torch.Tensor:
    return features / torch.norm(features, p=2, dim=-1, keepdim=True)


def display_label(label: str) -> str:
    return LABEL_ALIASES.get(str(label), str(label))


def display_image_path(image_path: str) -> str:
    # Ưu tiên ảnh gốc để frontend hiển thị đẹp hơn.
    if not image_path:
        return ""

    folder = os.path.dirname(image_path)
    for filename in ("1_original.png", "1.png"):
        candidate_path = os.path.join(folder, filename).replace("\\", "/")
        if os.path.exists(os.path.join(BASE_DIR, candidate_path)):
            return candidate_path

    return image_path


def encode_texts(prompts: list[str]) -> torch.Tensor:
    # Mã hóa text bằng CLIP.
    encoded_batches = []

    with torch.inference_mode():
        for start in range(0, len(prompts), TEXT_BATCH_SIZE):
            batch = prompts[start : start + TEXT_BATCH_SIZE]
            inputs = processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            outputs = model.text_model(**inputs)
            features = model.text_projection(outputs.pooler_output)
            encoded_batches.append(normalize(features).cpu())

    return torch.cat(encoded_batches, dim=0).to(device)


def encode_images(images: list[Image.Image]) -> torch.Tensor:
    # Mã hóa ảnh bằng CLIP.
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(outputs.pooler_output)
        return normalize(features)


def ensure_resources_loaded() -> None:
    # Kiểm tra tài nguyên đã sẵn sàng.
    resources = (model, processor, faiss_index, metadata_df, ood_text_feats, all_text_feats)
    if any(resource is None for resource in resources):
        raise HTTPException(status_code=503, detail="Server resources are still loading")


async def read_uploaded_image(file: UploadFile) -> Image.Image:
    # Đọc ảnh upload.
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc


def assert_traffic_sign(query_tensor: torch.Tensor) -> None:
    # Lọc ảnh không phải biển báo.
    similarities = torch.matmul(query_tensor, ood_text_feats.T)[0].cpu().numpy()
    print(
        f"OOD Sims: Traffic Sign = {similarities[0]:.4f}, "
        f"Random = {similarities[1]:.4f}"
    )

    if similarities[1] > similarities[0]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Hình ảnh không hợp lệ! Vui lòng tải lên đúng đoạn cắt có chứa biển báo."),
        )

def build_visual_candidates(query_vector, top_k: int) -> tuple[list[dict], list[int]]:
    # Tìm ảnh gần nhất bằng FAISS.
    distances, indices = faiss_index.search(query_vector, max(top_k, TOP_CANDIDATES))
    candidates = []
    candidate_indices = []

    for position, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        row = metadata_df.iloc[idx].fillna("").to_dict()
        candidate_indices.append(idx)
        candidates.append(
            {
                "idx": idx,
                "visual_score": float(distances[0][position]),
                "final_score": float(distances[0][position]),
                "row": row,
                "label": str(row.get("label", "Unknown")),
            }
        )

    return candidates, candidate_indices


def rerank_with_text(candidates: list[dict], candidate_indices: list[int], query_tensor: torch.Tensor) -> None:
    # Xếp hạng lại bằng text.
    if not candidate_indices:
        return

    candidate_text_features = all_text_feats[candidate_indices]
    text_similarities = torch.matmul(query_tensor, candidate_text_features.T)[0].cpu().numpy()

    for index, candidate in enumerate(candidates):
        text_score = float(text_similarities[index])
        candidate["final_score"] = candidate["visual_score"] + (text_score * TEXT_SCORE_WEIGHT)


def apply_mirror_direction_fix(
    candidates: list[dict],
    query_tensor: torch.Tensor,
    flipped_tensor: torch.Tensor,
) -> None:
    # Sửa nhầm trái/phải.
    candidate_labels = {candidate["label"] for candidate in candidates}

    for candidate in candidates:
        label = candidate["label"]
        mirror_label = MIRROR_PAIRS.get(label)
        if not mirror_label or mirror_label not in candidate_labels:
            continue

        reference_embedding = faiss_index.reconstruct(int(candidate["idx"]))
        reference_tensor = torch.from_numpy(reference_embedding).unsqueeze(0).to(device)
        original_similarity = torch.matmul(query_tensor, reference_tensor.T)[0].item()
        flipped_similarity = torch.matmul(flipped_tensor, reference_tensor.T)[0].item()

        if flipped_similarity > original_similarity:
            candidate["final_score"] -= MIRROR_WRONG_DIRECTION_PENALTY
            print(
                f"  FLIP-FIX: {label} penalized "
                f"(orig={original_similarity:.4f}, flip={flipped_similarity:.4f})"
            )
        else:
            candidate["final_score"] += MIRROR_CORRECT_DIRECTION_BONUS
            print(
                f"  FLIP-FIX: {label} confirmed "
                f"(orig={original_similarity:.4f}, flip={flipped_similarity:.4f})"
            )


def format_results(candidates: list[dict], top_k: int) -> list[dict]:
    # Tạo kết quả trả về.
    results = []
    seen_labels = set()

    for candidate in sorted(candidates, key=lambda item: item["final_score"], reverse=True):
        label = display_label(candidate["label"])
        if label in seen_labels:
            continue

        seen_labels.add(label)
        row = candidate["row"]
        display_score = min(candidate["final_score"] / DISPLAY_SCORE_DENOMINATOR, 1.0)
        results.append(
            {
                "rank": len(results) + 1,
                "score": display_score,
                "label": label,
                "group": str(row.get("group", "Unknown")),
                "meaning": str(row.get("meaning", "No meaning available")),
                "advice": str(row.get("advice", "Tuân thủ luật giao thông")),
                "image_path": display_image_path(str(row.get("image_path", ""))),
            }
        )

        if len(results) >= top_k:
            break

    return results


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
    # Load tài nguyên khi khởi động.
    global model, processor, faiss_index, metadata_df, ood_text_feats, all_text_feats

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    require_file(FAISS_INDEX_PATH, "FAISS index")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    print(f"Loading Metadata from {METADATA_PATH}...")
    require_file(METADATA_PATH, "Metadata file")
    metadata_df = pd.read_csv(METADATA_PATH)

    print("Pre-computing OOD text embeddings...")
    ood_text_feats = encode_texts(OOD_PROMPTS)

    print("Pre-computing text embeddings for all signs...")
    prompts = [
        f"A Vietnamese traffic sign that means: {str(row.get('meaning', ''))}"
        for _, row in metadata_df.iterrows()
    ]
    all_text_feats = encode_texts(prompts)
    print(f"Pre-computed {all_text_feats.shape[0]} text embeddings!")
    print("All resources loaded successfully!")


@app.get("/")
def read_root():
    return {"message": "Traffic Sign Recognition API is running!"}


@app.post("/search")
async def search_sign(file: UploadFile = File(...), top_k: int = 3):
    # API nhận diện biển báo.
    ensure_resources_loaded()
    image = await read_uploaded_image(file)
    top_k = max(1, min(int(top_k), 10))

    try:
        start_time = time.perf_counter()
        # Chuẩn hóa ảnh đầu vào.
        image_small = image.resize(IMAGE_SIZE, Image.LANCZOS)
        image_flipped = ImageOps.mirror(image_small)

        # Tạo vector ảnh.
        image_features = encode_images([image_small, image_flipped])
        query_vector = image_features[0:1].cpu().numpy()
        query_tensor = image_features[0:1]
        flipped_tensor = image_features[1:2]

        assert_traffic_sign(query_tensor)
        candidates, candidate_indices = build_visual_candidates(query_vector, top_k)
        rerank_with_text(candidates, candidate_indices, query_tensor)
        apply_mirror_direction_fix(candidates, query_tensor, flipped_tensor)
        results = format_results(candidates, top_k)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"  >> Total inference: {elapsed_ms:.0f}ms")
        return {"results": results}

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
