"""
Data Augmentation — sinh 18 biến thể cho mỗi biển báo.
Chạy: python backend/augment.py
"""

import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_aug")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

GROUP_FOLDERS = {
    "Prohibitory Signs": "cấm",
    "Warning Signs": "nguy hiểm",
    "Mandatory Signs": "hiệu lệnh",
    "Information Signs": "chỉ dẫn",
}

ROTATION_ANGLES = [10, -10, 20, -20, 30, -30, 45, -45]
PERSPECTIVE_OFFSETS = [(15, 10), (10, 15)]
SHEAR_ANGLES = [(5, "shear_5deg"), (-5, "shear_m5deg")]


def _save_if_new(path, image):
    if not os.path.exists(path):
        image.save(path)


def augment_image(img_path, save_dir):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return []

    base = os.path.splitext(os.path.basename(img_path))[0]
    w, h = img.size
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    files = []

    def add(name, image):
        _save_if_new(os.path.join(save_dir, name), image)
        files.append(name)

    add(f"{base}_original.png", img.copy())
    add(f"{base}_blur.png", img.filter(ImageFilter.GaussianBlur(radius=2)))
    add(f"{base}_bright.png", ImageEnhance.Brightness(img).enhance(1.5))
    add(f"{base}_dark.png", ImageEnhance.Brightness(img).enhance(0.5))

    for angle in ROTATION_ANGLES:
        name = f"{base}_rot{angle}.png"
        path = os.path.join(save_dir, name)
        if not os.path.exists(path):
            img.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC).save(path)
        files.append(name)

    for i, (dx, dy) in enumerate(PERSPECTIVE_OFFSETS, 1):
        name = f"{base}_persp{i}.png"
        path = os.path.join(save_dir, name)
        if not os.path.exists(path):
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([[dx, dy], [w - dx, 0], [0, h - dy], [w, h]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(img_cv, M, (w, h), borderValue=(255, 255, 255))
            cv2.imwrite(path, warped)
        files.append(name)

    for angle_deg, suffix in SHEAR_ANGLES:
        name = f"{base}_{suffix}.png"
        path = os.path.join(save_dir, name)
        if not os.path.exists(path):
            shear = np.tan(np.radians(angle_deg))
            M = np.float32([[1, shear, 0], [0, 1, 0]])
            new_w = int(w + abs(shear) * h)
            sheared = cv2.warpAffine(img_cv, M, (new_w, h), borderValue=(255, 255, 255))
            Image.fromarray(cv2.cvtColor(sheared, cv2.COLOR_BGR2RGB)).save(path)
        files.append(name)

    name = f"{base}_zoom_in.png"
    path = os.path.join(save_dir, name)
    if not os.path.exists(path):
        m = int(min(w, h) * 0.1)
        img.crop((m, m, w - m, h - m)).resize((w, h), Image.LANCZOS).save(path)
    files.append(name)

    name = f"{base}_zoom_out.png"
    path = os.path.join(save_dir, name)
    if not os.path.exists(path):
        small = img.resize((int(w * 0.7), int(h * 0.7)), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
        canvas.save(path)
    files.append(name)

    return files


def main():
    rows = []
    total_signs = 0

    for group_folder, group_vn in GROUP_FOLDERS.items():
        group_path = os.path.join(DATASET_DIR, group_folder)
        if not os.path.isdir(group_path):
            continue

        for label in sorted(os.listdir(group_path)):
            sign_dir = os.path.join(group_path, label)
            if not os.path.isdir(sign_dir):
                continue

            source = os.path.join(sign_dir, "1.png")
            if not os.path.exists(source):
                pngs = sorted(f for f in os.listdir(sign_dir) if f.lower().endswith(".png"))
                if not pngs:
                    continue
                source = os.path.join(sign_dir, pngs[0])

            augment_image(source, sign_dir)
            total_signs += 1

            for fname in sorted(os.listdir(sign_dir)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                rows.append({
                    "image_path": f"dataset_aug/{group_folder}/{label}/{fname}",
                    "group": group_vn,
                    "group_source": group_folder,
                    "label": label,
                    "meaning": "",
                    "class_id": "",
                    "advice": "",
                })

        print(f"  {group_folder}: xong")

    new_df = pd.DataFrame(rows)
    if os.path.exists(METADATA_PATH):
        old_info = pd.read_csv(METADATA_PATH).groupby("label").first()[["meaning", "advice"]].reset_index()
        new_df = new_df.drop(columns=["meaning", "advice"]).merge(old_info, on="label", how="left")
        new_df["meaning"] = new_df["meaning"].fillna("")
        new_df["advice"] = new_df["advice"].fillna("")

    new_df = new_df.drop_duplicates(subset=["image_path"])
    new_df.to_csv(METADATA_PATH, index=False)
    print(f"\nKết quả: {total_signs} biển, {len(new_df)} ảnh.")
    print("Tiếp theo: python backend/rebuild_faiss.py")


if __name__ == "__main__":
    main()
