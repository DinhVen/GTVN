"""
Generate weight-limit truck signs (P.106b variants) for different tonnage values.
Uses the existing P.106b sign as template and replaces the weight text.
"""
import os
import csv
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_aug", "Prohibitory Signs")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata.csv")

# Template sign
TEMPLATE_PATH = os.path.join(DATASET_DIR, "P.106b", "1.png")

# Weight values to generate (common QCVN 41 values)
# P.106b already has 2.5t, P.106c has 3.5t
WEIGHTS = ["1,5t", "2t", "5t", "7t", "8t", "10t", "15t"]


def create_weight_sign(weight_text):
    """
    Create a weight-limit truck sign by modifying the template.
    Opens the template P.106b, covers the old text area, writes new weight.
    """
    template = Image.open(TEMPLATE_PATH).convert("RGBA")
    size = template.size[0]  # 960
    draw = ImageDraw.Draw(template)
    
    # The text "2,5 t" on the original 960x960 image
    # Located on the truck body (black rectangle area, right side)
    text_area = (460, 380, 830, 580)  # left, top, right, bottom
    draw.rectangle(text_area, fill=(0, 0, 0, 255))
    
    # Draw new weight text
    font = None
    font_size = 120 if len(weight_text) <= 3 else 100
    
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except:
                continue
    
    if font is None:
        font = ImageFont.load_default()
    
    # Center text in the text area
    bbox = draw.textbbox((0, 0), weight_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    area_cx = (text_area[0] + text_area[2]) // 2
    area_cy = (text_area[1] + text_area[3]) // 2
    
    tx = area_cx - tw // 2
    ty = area_cy - th // 2
    
    draw.text((tx, ty), weight_text, fill=(255, 255, 255, 255), font=font)
    
    # Convert to RGB with white background
    bg = Image.new("RGB", template.size, (255, 255, 255))
    bg.paste(template, mask=template.split()[3])
    return bg


def generate_rotation_augmentations(img, folder):
    """Generate rotation augmentations."""
    angles = [-20, -30, -45, 20, 30, 45]
    for angle in angles:
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
        rotated.save(os.path.join(folder, f"1_rot{angle}.png"))


def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if not os.path.exists(TEMPLATE_PATH):
        print(f"ERROR: Template not found: {TEMPLATE_PATH}")
        return
    
    # Read existing metadata
    rows = []
    existing_labels = set()
    with open(METADATA_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
            existing_labels.add(row["label"])
    
    new_count = 0
    
    for weight in WEIGHTS:
        # Convert weight text to label-safe format: "1,5t" -> "1.5"
        weight_num = weight.replace(",", ".").replace("t", "")
        label = f"P.106b_{weight_num}"
        
        if label in existing_labels:
            print(f"  SKIP {label} (already exists)")
            continue
        
        # Create folder
        folder = os.path.join(DATASET_DIR, label)
        os.makedirs(folder, exist_ok=True)
        
        # Generate sign image
        img = create_weight_sign(weight)
        img.save(os.path.join(folder, "1.png"), "PNG")
        
        # Generate augmentations
        generate_rotation_augmentations(img, folder)
        
        # Add metadata entries
        rel_base = f"dataset_aug/Prohibitory Signs/{label}"
        weight_display = weight.replace(",", ",")  # Keep Vietnamese comma format
        meaning = f"Cấm ô tô tải trên {weight_display}"
        advice = f"Xe tải có tải trọng trên {weight_display} không được đi vào đoạn đường này."
        
        rows.append({
            "label": label, "group": "Biển cấm",
            "meaning": meaning, "advice": advice,
            "image_path": f"{rel_base}/1.png"
        })
        for angle in [-20, -30, -45, 20, 30, 45]:
            rows.append({
                "label": label, "group": "Biển cấm",
                "meaning": meaning, "advice": advice,
                "image_path": f"{rel_base}/1_rot{angle}.png"
            })
        
        new_count += 1
        print(f"  CREATED {label} -> Cấm xe tải trên {weight_display}")
    
    # Write updated metadata
    with open(METADATA_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nDone! Created {new_count} new weight signs.")
    print(f"Total metadata rows: {len(rows)}")
    print(f"\nNOTE: Run rebuild_faiss.py to update the FAISS index!")


if __name__ == "__main__":
    main()
