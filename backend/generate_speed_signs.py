"""
Generate speed limit sign images for the dataset.
Creates clean P.127 signs with different speed values (20-120 km/h).
Each sign gets its own folder + augmented rotations + metadata entry.
"""
import os
import sys
import csv
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_aug", "Prohibitory Signs")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata.csv")

# Speed values to generate (common Vietnamese speed limits)
SPEEDS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120]

def draw_speed_sign(speed, size=512):
    """Draw a Vietnamese P.127 speed limit sign."""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    outer_r = size // 2 - 10
    inner_r = int(outer_r * 0.82)
    
    # Red outer circle
    draw.ellipse(
        [center - outer_r, center - outer_r, center + outer_r, center + outer_r],
        fill=(220, 20, 20), outline=(180, 0, 0), width=3
    )
    
    # White inner circle
    draw.ellipse(
        [center - inner_r, center - inner_r, center + inner_r, center + inner_r],
        fill=(255, 255, 255), outline=(255, 255, 255)
    )
    
    # Speed number text
    text = str(speed)
    
    # Try to use a bold font, fall back to default
    font = None
    font_size = int(size * 0.38) if len(text) <= 2 else int(size * 0.30)
    
    # Try common Windows fonts
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf", 
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/impact.ttf",
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
    
    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = center - tw // 2
    ty = center - th // 2 - int(size * 0.02)
    
    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
    
    # Convert to RGB (white background)
    bg = Image.new("RGB", (size, size), (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def generate_rotation_augmentations(img, folder):
    """Generate rotation augmentations like the existing dataset."""
    angles = [-20, -30, -45, 20, 30, 45]
    for angle in angles:
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
        rot_name = f"1_rot{angle}.png"
        rotated.save(os.path.join(folder, rot_name))


def main():
    # Read existing metadata
    existing_labels = set()
    rows = []
    with open(METADATA_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
            existing_labels.add(row["label"])
    
    new_count = 0
    
    for speed in SPEEDS:
        label = f"P.127_{speed}"
        
        if label in existing_labels:
            print(f"  SKIP {label} (already exists)")
            continue
        
        # Create folder
        folder = os.path.join(DATASET_DIR, label)
        os.makedirs(folder, exist_ok=True)
        
        # Generate sign image
        img = draw_speed_sign(speed)
        img_path = os.path.join(folder, "1.png")
        img.save(img_path, "PNG")
        
        # Generate augmentations
        generate_rotation_augmentations(img, folder)
        
        # Add metadata entries (original + augmented)
        rel_base = f"dataset_aug/Prohibitory Signs/{label}"
        
        meaning = f"Giới hạn tốc độ tối đa {speed} km/h"
        advice = f"Không được chạy quá {speed} km/h. Vi phạm sẽ bị xử phạt theo Nghị định 100."
        
        # Main image
        rows.append({
            "label": label,
            "group": "Biển cấm",
            "meaning": meaning,
            "advice": advice,
            "image_path": f"{rel_base}/1.png"
        })
        
        # Augmented images
        for angle in [-20, -30, -45, 20, 30, 45]:
            rows.append({
                "label": label,
                "group": "Biển cấm",
                "meaning": meaning,
                "advice": advice,
                "image_path": f"{rel_base}/1_rot{angle}.png"
            })
        
        new_count += 1
        print(f"  CREATED {label} -> {speed} km/h ({folder})")
    
    # Write updated metadata
    with open(METADATA_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nDone! Created {new_count} new speed signs.")
    print(f"Total metadata rows: {len(rows)}")
    print(f"\nNOTE: Run rebuild_faiss.py to update the FAISS index!")


if __name__ == "__main__":
    main()
