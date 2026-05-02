"""
Script tự động tạo thêm ảnh xoay góc lớn cho dataset_aug
Thêm các góc: ±20°, ±30°, ±45° cho mỗi ảnh gốc (1.png)
Cập nhật metadata.csv sau khi sinh ảnh
"""

import os
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_aug")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

# Góc xoay mới cần bổ sung (các góc hiện tại: rot10, rot-10)
NEW_ROTATION_ANGLES = [-45, -30, -20, 20, 30, 45]

def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Xoay ảnh theo góc chỉ định, giữ nguyên kích thước gốc (crop trắng bị fill màu nền)."""
    # Dùng expand=False để giữ nguyên kích thước, vùng trống fill màu trắng
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    return rotated

def get_angle_suffix(angle: float) -> str:
    if angle > 0:
        return f"rot{int(angle)}"
    else:
        return f"rot{int(angle)}"  # sẽ ra rot-20, rot-30, rot-45

def main():
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded metadata: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Danh sách suffix đã có trong metadata để tránh tạo trùng
    existing_suffixes = set()
    for path in df['image_path'].tolist():
        filename = os.path.basename(str(path))
        name_no_ext = os.path.splitext(filename)[0]
        if '_' in name_no_ext:
            suffix = name_no_ext.split('_', 1)[1]
            existing_suffixes.add(suffix)

    print(f"Existing augmentation types: {sorted(existing_suffixes)}")

    new_rows = []
    created_count = 0
    skipped_count = 0

    # Lọc chỉ lấy ảnh gốc 1.png để tạo rotation từ đó
    original_rows = df[df['image_path'].apply(
        lambda p: os.path.basename(str(p)) == '1.png' or os.path.basename(str(p)) == '1_original.png'
    )]

    print(f"\nFound {len(original_rows)} original base images (1.png) to augment from.")
    print(f"Will create {len(NEW_ROTATION_ANGLES)} new rotation angles per image: {NEW_ROTATION_ANGLES}")
    print(f"Estimated new images: ~{len(original_rows) * len(NEW_ROTATION_ANGLES)}")
    print()

    for _, row in tqdm(original_rows.iterrows(), total=len(original_rows), desc="Generating rotations"):
        img_relative_path = str(row['image_path'])
        img_abs_path = os.path.join(BASE_DIR, img_relative_path)

        if not os.path.exists(img_abs_path):
            skipped_count += 1
            continue

        try:
            image = Image.open(img_abs_path).convert("RGB")
        except Exception as e:
            print(f"Cannot open {img_abs_path}: {e}")
            skipped_count += 1
            continue

        img_dir = os.path.dirname(img_abs_path)

        for angle in NEW_ROTATION_ANGLES:
            suffix = get_angle_suffix(angle)
            new_filename = f"1_{suffix}.png"
            new_abs_path = os.path.join(img_dir, new_filename)

            # Bỏ qua nếu file đã tồn tại
            if os.path.exists(new_abs_path):
                skipped_count += 1
                continue

            # Tạo ảnh xoay
            rotated = rotate_image(image, angle)
            rotated.save(new_abs_path)
            created_count += 1

            # Tính relative path để lưu vào metadata (dùng / cho cross-platform)
            new_relative_path = os.path.relpath(new_abs_path, BASE_DIR).replace("\\", "/")

            # Tạo row mới trong metadata
            new_row = row.to_dict()
            new_row['image_path'] = new_relative_path
            new_rows.append(new_row)

    print(f"\nDone!")
    print(f"  Created: {created_count} new images")
    print(f"  Skipped (already existed): {skipped_count}")
    print(f"  New metadata rows to add: {len(new_rows)}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_df.to_csv(METADATA_PATH, index=False)
        print(f"\nMetadata updated: {len(df)} -> {len(updated_df)} rows")
        print(f"Saved to: {METADATA_PATH}")
    else:
        print("\nNo new rows to add (all augmentations already exist).")

    print("\nDone! Now run rebuild_faiss.py to re-index the new images.")

if __name__ == "__main__":
    main()
