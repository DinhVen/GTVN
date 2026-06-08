import os
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset_aug"
DATA_DIR = BASE_DIR / "data"
METADATA_PATH = DATA_DIR / "metadata.csv"

GROUPS = {
    "Biển cấm": "cấm",
    "Biển nguy hiểm": "nguy hiểm",
    "Biển hiệu lệnh": "hiệu lệnh",
    "Biển chỉ dẫn": "chỉ dẫn",
    "Biển phụ": "biển phụ",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_old_descriptions() -> pd.DataFrame | None:
    """Lấy lại meaning/advice cũ nếu đã có metadata."""
    if not METADATA_PATH.exists():
        return None

    old_df = pd.read_csv(METADATA_PATH).fillna("")
    required_cols = {"label", "meaning", "advice"}
    if not required_cols.issubset(old_df.columns):
        return None

    return (
        old_df.groupby("label", as_index=False)
        .first()[["label", "meaning", "advice"]]
    )


def build_rows() -> list[dict]:
    """Duyệt dataset_aug và tạo từng dòng metadata."""
    rows = []

    for folder_name, group_name in GROUPS.items():
        group_dir = DATASET_DIR / folder_name
        if not group_dir.is_dir():
            print(f"Bỏ qua nhóm không tồn tại: {group_dir}")
            continue

        for sign_dir in sorted(group_dir.iterdir()):
            if not sign_dir.is_dir():
                continue

            label = sign_dir.name
            for image_path in sorted(sign_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                relative_path = image_path.relative_to(BASE_DIR).as_posix()
                rows.append(
                    {
                        "image_path": relative_path,
                        "group": group_name,
                        "group_source": folder_name,
                        "label": label,
                        "class_id": "",
                        "meaning": "",
                        "advice": "",
                    }
                )

    return rows


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    rows = build_rows()
    if not rows:
        raise RuntimeError("Không tìm thấy ảnh nào trong dataset_aug.")

    metadata = pd.DataFrame(rows).drop_duplicates(subset=["image_path"])

    old_descriptions = load_old_descriptions()
    if old_descriptions is not None:
        metadata = metadata.drop(columns=["meaning", "advice"]).merge(
            old_descriptions,
            on="label",
            how="left",
        )
        metadata["meaning"] = metadata["meaning"].fillna("")
        metadata["advice"] = metadata["advice"].fillna("")

    metadata = metadata[
        ["image_path", "group", "group_source", "label", "class_id", "meaning", "advice"]
    ]
    metadata.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

    print(f"Đã tạo metadata: {METADATA_PATH}")
    print(f"Số dòng ảnh: {len(metadata)}")
    print(f"Số mã biển: {metadata['label'].nunique()}")
    print("Bước tiếp theo: python backend/rebuild_faiss.py")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
