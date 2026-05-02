"""
AUDIT FIX ROUND 4: Sửa tất cả I.444g-m, I.445c-h
"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv("d:/gtvn/data/metadata.csv", encoding="utf-8-sig")

fixes = {
    "I.444g": (
        "Chỉ dẫn di tích, chùa chiền (danh thắng)",
        "Đi theo hướng mũi tên để đến di tích / chùa chiền."
    ),
    "I.444h": (
        "Chỉ dẫn trạm xăng dầu (Gasoline)",
        "Đi theo hướng mũi tên để đến trạm xăng dầu."
    ),
    "I.444i": (
        "Chỉ dẫn tiệm rửa xe (Car Wash)",
        "Đi theo hướng mũi tên để đến tiệm rửa xe."
    ),
    "I.444j": (
        "Chỉ dẫn bến phà (Ferry)",
        "Đi theo hướng mũi tên để đến bến phà."
    ),
    "I.444k": (
        "Chỉ dẫn ga tàu điện ngầm (Metro)",
        "Đi theo hướng mũi tên để đến ga tàu điện ngầm."
    ),
    "I.444l": (
        "Chỉ dẫn nhà hàng ăn uống (Restaurant)",
        "Đi theo hướng mũi tên để đến nhà hàng ăn uống."
    ),
    "I.444m": (
        "Chỉ dẫn trạm sửa chữa ô tô (Garage)",
        "Đi theo hướng mũi tên để đến trạm sửa chữa ô tô."
    ),
    "I.445c": (
        "Đi chậm, đường nhiều sương mù",
        "Phía trước có sương mù dày, giảm tốc độ và bật đèn sương mù."
    ),
    "I.445d": (
        "Nền đường yếu",
        "Đường phía trước nền yếu, giảm tốc độ để tránh hư hại."
    ),
    "I.445e": (
        "Xe lớn đi sát về bên phải",
        "Xe tải, xe lớn phải đi sát lề bên phải để nhường đường."
    ),
    "I.445f": (
        "Chú ý gió ngang mạnh",
        "Phía trước có gió ngang mạnh, giữ vững tay lái."
    ),
    "I.445g": (
        "Đoạn đường hay xảy ra tai nạn",
        "Khu vực nguy hiểm, thường xảy ra tai nạn, tập trung quan sát."
    ),
    "I.445h": (
        "Xuống dốc liên tục",
        "Phía trước xuống dốc liên tục, sử dụng số thấp, không đạp phanh liên tục."
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
print(f"\nDone! Fixed {count} labels. Total I-group should now be correct.")
