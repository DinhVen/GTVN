"""
AUDIT FIX ROUND 2: Sửa thêm các biển I còn sai sau đợt 1
"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv("d:/gtvn/data/metadata.csv", encoding="utf-8-sig")

fixes = {
    "I.433b": (
        "Khu cắm trại (lều)",
        "Phía trước có khu cắm trại dạng lều, có thể dừng nghỉ ngơi."
    ),
    "I.433c": (
        "Bãi đỗ xe kéo di động (caravan)",
        "Phía trước có bãi đỗ dành cho xe kéo di động (caravan)."
    ),
    "I.433d": (
        "Khu cắm trại và đỗ xe kéo di động",
        "Phía trước có khu cắm trại kết hợp bãi đỗ xe caravan."
    ),
    "I.433e": (
        "Nhà nghỉ rừng / cabin nghỉ dưỡng",
        "Phía trước có nhà nghỉ dạng cabin giữa rừng."
    ),
    "I.434b": (
        "Bến xe tải",
        "Phía trước có bến xe tải, xe tải lên xuống hàng tại đây."
    ),
    "I.441a": (
        "Báo trước công trường phía trước 500m",
        "Phía trước 500m có công trường thi công, giảm tốc độ và chú ý."
    ),
    "I.441b": (
        "Báo trước công trường phía trước 100m",
        "Phía trước 100m có công trường thi công, giảm tốc độ."
    ),
    "I.441c": (
        "Báo trước công trường phía trước 50m",
        "Phía trước 50m có công trường thi công, đi chậm và quan sát."
    ),
    "I.442": (
        "Chợ / khu mua sắm (Market)",
        "Phía trước có chợ hoặc khu mua sắm, chú ý người qua đường."
    ),
    "I.443": (
        "Biển chú ý, cảnh giác (tam giác vàng)",
        "Biển cảnh báo chung yêu cầu tài xế tập trung chú ý."
    ),
    "I.444a": (
        "Ga tàu hỏa (Railway Station)",
        "Phía trước có ga tàu hỏa, đi theo hướng mũi tên."
    ),
    "I.445a": (
        "Đường vòng xuyến / chạy chậm",
        "Phía trước có vòng xuyến, giảm tốc độ và nhường đường xe trong vòng xuyến."
    ),
    "I.446": (
        "Đường / lối đi dành cho người khuyết tật",
        "Khu vực dành cho người khuyết tật, nhường đường và hỗ trợ khi cần."
    ),
    "I.448": (
        "Đường cứu nạn (thoát hiểm)",
        "Phía trước 300m có đường cứu nạn, sử dụng khi mất phanh hoặc khẩn cấp."
    ),
    "I.449": (
        "Số hiệu đường quốc tế (Asian Highway)",
        "Đường thuộc mạng lưới đường bộ xuyên Á (Asian Highway Network)."
    ),
}

count_fixed = 0
for label, (new_meaning, new_advice) in fixes.items():
    mask = df["label"] == label
    n = mask.sum()
    if n > 0:
        old = df.loc[mask, "meaning"].iloc[0]
        df.loc[mask, "meaning"] = new_meaning
        df.loc[mask, "advice"] = new_advice
        print(f"  FIX {label} ({n} rows): '{old}' -> '{new_meaning}'")
        count_fixed += 1

# Kiểm tra thêm I.417c nếu tồn tại
mask_417c = df["label"] == "I.417c"
if mask_417c.sum() > 0:
    print(f"  NOTE: I.417c exists ({mask_417c.sum()} rows), meaning: '{df.loc[mask_417c, 'meaning'].iloc[0]}'")

df.to_csv("d:/gtvn/data/metadata.csv", index=False, encoding="utf-8-sig")
print(f"\nDone! Fixed {count_fixed} more labels in metadata.csv")
