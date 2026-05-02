import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv("d:/gtvn/data/metadata.csv", encoding="utf-8-sig")

fixes = {
    "I.423a": {
        "meaning": "Đường cho người đi bộ sang ngang (người đi từ trái sang)",
        "advice": "Giảm tốc độ, quan sát và nhường đường cho người đi bộ tại vạch kẻ ngang."
    },
    "I.423b": {
        "meaning": "Đường cho người đi bộ sang ngang (người đi từ phải sang)",
        "advice": "Giảm tốc độ, quan sát và nhường đường cho người đi bộ tại vạch kẻ ngang."
    },
    "I.423c": {
        "meaning": "Đường cho người đi bộ sang ngang (cả hai hướng)",
        "advice": "Giảm tốc độ, chú ý người đi bộ từ hai phía, sẵn sàng dừng xe nhường đường."
    },
}

for label, info in fixes.items():
    mask = df["label"] == label
    count = mask.sum()
    old_meaning = df.loc[mask, "meaning"].iloc[0] if count > 0 else "N/A"
    df.loc[mask, "meaning"] = info["meaning"]
    df.loc[mask, "advice"] = info["advice"]
    print(f"  {label}: '{old_meaning}' -> '{info['meaning']}' ({count} rows)")

df.to_csv("d:/gtvn/data/metadata.csv", index=False, encoding="utf-8-sig")
print("\nDone! metadata.csv updated.")
