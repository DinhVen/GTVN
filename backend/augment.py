"""
Script tăng cường dữ liệu (Data Augmentation) cho TẤT CẢ biển báo.
Đọc ảnh gốc 1.png trong mỗi thư mục biển → sinh 18 biến thể.
Thư viện: Pillow (xoay, mờ, sáng/tối, zoom) + OpenCV (phối cảnh, xô lệch).
"""
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_aug")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

# 4 nhóm biển và tên thư mục tương ứng
GROUP_FOLDERS = {
    "Prohibitory Signs": "cấm",
    "Warning Signs": "nguy hiểm",
    "Mandatory Signs": "hiệu lệnh",
    "Information Signs": "chỉ dẫn",
}


def augment_image(img_path, save_dir):
    """Sinh 18 biến thể từ 1 ảnh gốc, lưu cùng thư mục."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return []

    base = os.path.splitext(os.path.basename(img_path))[0]
    w, h = img.size
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    new_files = []

    def save_if_new(name, image):
        path = os.path.join(save_dir, name)
        if not os.path.exists(path):
            image.save(path)
        new_files.append(name)

    # 1. Bản sao gốc
    save_if_new(f"{base}_original.png", img.copy())

    # 2. Làm mờ — mô phỏng camera rung, ảnh không nét
    save_if_new(f"{base}_blur.png", img.filter(ImageFilter.GaussianBlur(radius=2)))

    # 3. Tăng sáng — mô phỏng trời nắng gắt
    save_if_new(f"{base}_bright.png", ImageEnhance.Brightness(img).enhance(1.5))

    # 4. Giảm sáng — mô phỏng ban đêm, bóng râm
    save_if_new(f"{base}_dark.png", ImageEnhance.Brightness(img).enhance(0.5))

    # 5-12. Xoay ±10°, ±20°, ±30°, ±45° — mô phỏng biển bị nghiêng
    for angle in [10, -10, 20, -20, 30, -30, 45, -45]:
        name = f"{base}_rot{angle}.png"
        path = os.path.join(save_dir, name)
        if not os.path.exists(path):
            img.rotate(angle, expand=True, fillcolor=(255, 255, 255),
                       resample=Image.BICUBIC).save(path)
        new_files.append(name)

    # 13-14. Biến dạng phối cảnh — mô phỏng nhìn từ góc lệch
    for i, (dx, dy) in enumerate([(15, 10), (10, 15)], 1):
        name = f"{base}_persp{i}.png"
        path = os.path.join(save_dir, name)
        if not os.path.exists(path):
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([[dx, dy], [w - dx, 0], [0, h - dy], [w, h]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(img_cv, M, (w, h),
                                         borderValue=(255, 255, 255))
            cv2.imwrite(path, warped)
        new_files.append(name)

    # 15-16. Xô lệch ±5° — mô phỏng ảnh bị méo
    for angle_deg, suffix in [(5, "shear_5deg"), (-5, "shear_m5deg")]:
        name = f"{base}_{suffix}.png"
        path = os.path.join(save_dir, name)
        if not os.path.exists(path):
            shear = np.tan(np.radians(angle_deg))
            M = np.float32([[1, shear, 0], [0, 1, 0]])
            new_w = int(w + abs(shear) * h)
            sheared = cv2.warpAffine(img_cv, M, (new_w, h),
                                     borderValue=(255, 255, 255))
            Image.fromarray(cv2.cvtColor(sheared, cv2.COLOR_BGR2RGB)).save(path)
        new_files.append(name)

    # 17. Zoom vào — mô phỏng chụp gần
    name = f"{base}_zoom_in.png"
    path = os.path.join(save_dir, name)
    if not os.path.exists(path):
        m = int(min(w, h) * 0.1)
        img.crop((m, m, w - m, h - m)).resize((w, h), Image.LANCZOS).save(path)
    new_files.append(name)

    # 18. Zoom ra — mô phỏng chụp xa
    name = f"{base}_zoom_out.png"
    path = os.path.join(save_dir, name)
    if not os.path.exists(path):
        small = img.resize((int(w * 0.7), int(h * 0.7)), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
        canvas.save(path)
    new_files.append(name)

    return new_files


def main():
    rows = []
    total_signs = 0
    total_images = 0

    # Duyệt tất cả thư mục nhóm biển
    for group_folder, group_vn in GROUP_FOLDERS.items():
        group_path = os.path.join(DATASET_DIR, group_folder)
        if not os.path.isdir(group_path):
            continue

        # Duyệt từng thư mục biển (P.102, W.201a...)
        labels = sorted(os.listdir(group_path))
        for label in labels:
            sign_dir = os.path.join(group_path, label)
            if not os.path.isdir(sign_dir):
                continue

            # Tìm ảnh gốc 1.png
            source_img = os.path.join(sign_dir, "1.png")
            if not os.path.exists(source_img):
                # Thử tìm file ảnh đầu tiên
                pngs = sorted(f for f in os.listdir(sign_dir)
                              if f.lower().endswith(".png"))
                if not pngs:
                    continue
                source_img = os.path.join(sign_dir, pngs[0])

            # Augmentation
            new_files = augment_image(source_img, sign_dir)
            total_signs += 1

            # Thêm tất cả ảnh trong thư mục vào danh sách metadata
            all_files = sorted(f for f in os.listdir(sign_dir)
                               if f.lower().endswith((".png", ".jpg", ".jpeg")))
            for fname in all_files:
                rel_path = os.path.join(
                    "dataset_aug", group_folder, label, fname
                ).replace("\\", "/")
                rows.append({
                    "image_path": rel_path,
                    "group": group_vn,
                    "group_source": group_folder,
                    "label": label,
                    "meaning": "",   # Cần điền sau
                    "class_id": "",
                    "advice": "",    # Cần điền sau
                })
                total_images += 1

        print(f"  {group_folder}: xong")

    # Tạo metadata mới từ scan thư mục
    new_df = pd.DataFrame(rows)

    # Ghép meaning + advice từ metadata cũ (nếu có)
    if os.path.exists(METADATA_PATH):
        old_df = pd.read_csv(METADATA_PATH)
        # Lấy meaning + advice theo label (mỗi label 1 dòng)
        label_info = old_df.groupby("label").first()[["meaning", "advice"]].reset_index()
        new_df = new_df.drop(columns=["meaning", "advice"])
        new_df = new_df.merge(label_info, on="label", how="left")
        new_df["meaning"] = new_df["meaning"].fillna("")
        new_df["advice"] = new_df["advice"].fillna("")

    # Loại bỏ ảnh trùng
    new_df = new_df.drop_duplicates(subset=["image_path"])

    # Lưu metadata
    new_df.to_csv(METADATA_PATH, index=False)
    print(f"\nKết quả: {total_signs} biển, {len(new_df)} ảnh trong metadata.")
    print("Tiếp theo: python backend/rebuild_faiss.py")


if __name__ == "__main__":
    main()
