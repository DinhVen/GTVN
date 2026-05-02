"""
AUDIT FIX ROUND 3: Sửa I.417c, I.444b-f, I.445b và các biển còn lại
"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv("d:/gtvn/data/metadata.csv", encoding="utf-8-sig")

fixes = {
    "I.417c": (
        "Chỉ dẫn hướng đi cho xe đạp và người đi bộ",
        "Xe đạp và người đi bộ đi theo hướng mũi tên trên biển."
    ),
    "I.444b": (
        "Chỉ dẫn sân bay (Airport)",
        "Đi theo hướng mũi tên để đến sân bay."
    ),
    "I.444c": (
        "Chỉ dẫn bãi đỗ xe (Parking Lot)",
        "Đi theo hướng mũi tên để đến bãi đỗ xe."
    ),
    "I.444d": (
        "Chỉ dẫn bến xe khách (Bus Station)",
        "Đi theo hướng mũi tên để đến bến xe khách."
    ),
    "I.444e": (
        "Chỉ dẫn trạm cấp cứu (First Aid)",
        "Đi theo hướng mũi tên để đến trạm cấp cứu gần nhất."
    ),
    "I.444f": (
        "Chỉ dẫn bến tàu thủy (Waterway Station)",
        "Đi theo hướng mũi tên để đến bến tàu thủy."
    ),
    "I.445b": (
        "Đường dốc, chạy chậm",
        "Phía trước có đường dốc, giảm tốc độ và sử dụng số thấp."
    ),
}

count = 0
for label, (new_meaning, new_advice) in fixes.items():
    mask = df["label"] == label
    n = mask.sum()
    if n > 0:
        old = df.loc[mask, "meaning"].iloc[0]
        df.loc[mask, "meaning"] = new_meaning
        df.loc[mask, "advice"] = new_advice
        print(f"  FIX {label} ({n} rows): '{old}' -> '{new_meaning}'")
        count += 1

df.to_csv("d:/gtvn/data/metadata.csv", index=False, encoding="utf-8-sig")
print(f"\nDone! Fixed {count} more labels. Checking remaining I.444-I.449...")

# Verify remaining
df2 = pd.read_csv("d:/gtvn/data/metadata.csv", encoding="utf-8-sig")
remaining = df2[df2['label'].str.startswith('I.44')][['label','meaning']].drop_duplicates('label').sort_values('label')
print(remaining.to_string(index=False))
