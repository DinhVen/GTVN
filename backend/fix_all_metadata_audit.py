"""
FULL AUDIT FIX: Sửa metadata cho tất cả biển I bị lệch ảnh vs meaning.
Nguyên tắc: meaning phải MÔ TẢ ĐÚNG ẢNH TRONG DATASET, không phải mã QCVN chuẩn.
"""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv("d:/gtvn/data/metadata.csv", encoding="utf-8-sig")

# Dict: label -> (meaning_moi, advice_moi)
# Chỉ sửa những biển BỊ SAI (đã xác minh bằng ảnh thực tế)
fixes = {
    # === I.415 -> ảnh "Hà Nội 29km" (biển xanh lá mũi tên) ===
    "I.415": (
        "Chỉ dẫn hướng và khoảng cách (biển xanh lá)",
        "Biển cho biết khoảng cách và hướng đến địa danh phía trước."
    ),
    # === I.416 -> ảnh sơ đồ đường vòng tránh Hòa Bình ===
    "I.416": (
        "Sơ đồ đường vòng tránh",
        "Chỉ dẫn đường vòng tránh qua khu vực, tuân theo hướng dẫn trên biển."
    ),
    # === I.417a -> ảnh "Hà Nội, xe trên 10t" sang phải ===
    "I.417a": (
        "Chỉ dẫn hướng đi cho xe tải (rẽ phải)",
        "Xe tải và xe kéo moóc đi theo hướng mũi tên rẽ phải."
    ),
    # === I.417b -> ảnh "Hà Nội, xe trên 10t" đi thẳng ===
    "I.417b": (
        "Chỉ dẫn hướng đi cho xe tải (đi thẳng)",
        "Xe tải và xe kéo moóc đi theo hướng mũi tên đi thẳng."
    ),
    # === I.417c -> cần kiểm tra thêm ===

    # === I.418 -> file ảnh bị hỏng (447 bytes null) ===
    # Giữ nguyên, ghi chú file hỏng

    # === I.419a -> ảnh "ĐỊA PHẬN HÀ NAM Km 215+700" ===
    "I.419a": (
        "Địa giới hành chính (tỉnh/thành phố)",
        "Bạn đang đi vào địa phận tỉnh/thành phố được ghi trên biển."
    ),
    # === I.419b -> ảnh "ĐỊA PHẬN HÀ NAM ZONE" ===
    "I.419b": (
        "Địa giới hành chính (song ngữ)",
        "Bạn đang đi vào địa phận tỉnh/thành phố, biển ghi song ngữ Việt-Anh."
    ),

    # === I.422a -> ảnh "VỊNH HẠ LONG" ===
    "I.422a": (
        "Địa danh du lịch",
        "Biển chỉ dẫn đến danh lam thắng cảnh du lịch nổi tiếng."
    ),
    # === I.422b -> ảnh "VỊNH HẠ LONG BAY" ===
    "I.422b": (
        "Địa danh du lịch (song ngữ)",
        "Biển chỉ dẫn đến danh lam thắng cảnh, ghi song ngữ Việt-Anh."
    ),

    # I.423a/b/c đã sửa ở bước trước (người đi bộ sang ngang)

    # === I.424a -> ảnh cầu thang (người leo lên bên phải) ===
    "I.424a": (
        "Cầu vượt đi bộ (bên phải)",
        "Người đi bộ sử dụng cầu vượt bộ hành bên phải để sang đường an toàn."
    ),
    # === I.424b -> ảnh cầu thang (người leo lên bên trái) ===
    "I.424b": (
        "Cầu vượt đi bộ (bên trái)",
        "Người đi bộ sử dụng cầu vượt bộ hành bên trái để sang đường an toàn."
    ),
    # === I.424c -> ảnh cầu thang (đi xuống bên phải) ===
    "I.424c": (
        "Hầm đi bộ (bên phải)",
        "Người đi bộ sử dụng hầm bộ hành bên phải để sang đường an toàn."
    ),
    # === I.424d -> ảnh cầu thang (đi xuống bên trái) ===
    "I.424d": (
        "Hầm đi bộ (bên trái)",
        "Người đi bộ sử dụng hầm bộ hành bên trái để sang đường an toàn."
    ),

    # === I.425 -> ảnh giường + dấu thập đỏ ===
    "I.425": (
        "Bệnh viện, trạm cấp cứu",
        "Phía trước có bệnh viện hoặc trạm y tế, có thể dừng cấp cứu nếu cần."
    ),

    # === I.427a -> ảnh tuốc nơ vít/cờ lê ===
    # Đã đúng: Trạm sửa chữa ô tô

    # === I.427b -> ảnh "TRẠM KTTT" ===
    "I.427b": (
        "Trạm kiểm tra tải trọng xe",
        "Phía trước có trạm KTTT, xe quá tải phải dừng kiểm tra."
    ),

    # === I.428a -> ảnh trạm xăng (cây xăng) ===
    "I.428a": (
        "Trạm xăng dầu",
        "Phía trước có trạm xăng dầu, có thể dừng đổ xăng."
    ),

    # === I.430 -> ảnh điện thoại + số 1900... ===
    "I.430": (
        "Điện thoại khẩn cấp / cứu hộ",
        "Biển ghi số điện thoại đường dây nóng cứu hộ, gọi khi cần hỗ trợ."
    ),

    # === I.431 -> ảnh "TRẠM DỪNG NGHỈ" (đồ ăn + xăng) ===
    "I.431": (
        "Trạm dừng nghỉ",
        "Phía trước có trạm dừng nghỉ, có dịch vụ ăn uống và đổ xăng."
    ),

    # === I.432 -> ảnh "KHÁCH SẠN" (giường) ===
    "I.432": (
        "Khách sạn, nhà nghỉ",
        "Phía trước có khách sạn hoặc nhà nghỉ, có thể dừng nghỉ ngơi."
    ),

    # === I.433a -> ảnh "NGHỈ MÁT" (cây thông) ===
    "I.433a": (
        "Khu nghỉ mát, cắm trại",
        "Phía trước có khu nghỉ mát hoặc bãi cắm trại."
    ),

    # === I.434a -> ảnh "BẾN XE BUÝT / BUS STOP" ===
    "I.434a": (
        "Bến xe buýt (Bus Stop)",
        "Phía trước có bến xe buýt, hành khách lên xuống tại đây."
    ),

    # === I.435 -> ảnh "BẾN XE ĐIỆN / TRAMWAY STATION" ===
    "I.435": (
        "Bến xe điện (Tramway Station)",
        "Phía trước có bến xe điện, hành khách lên xuống tại đây."
    ),

    # === I.436 -> ảnh "CSGT / POLICE" ===
    "I.436": (
        "Trạm cảnh sát giao thông (CSGT)",
        "Phía trước có trạm CSGT, giảm tốc độ và chấp hành luật giao thông."
    ),

    # === I.437 -> ảnh đường cao tốc (nền xanh lá) ===
    "I.437": (
        "Bắt đầu đường cao tốc",
        "Bạn đang vào đường cao tốc, tuân thủ tốc độ tối thiểu và tối đa."
    ),

    # === I.439 -> ảnh "CẦU YÊN CHÂU Km 252" ===
    "I.439": (
        "Tên cầu và lý trình",
        "Biển báo tên cầu và vị trí km trên tuyến đường."
    ),

    # === I.440 -> ảnh "ĐOẠN ĐƯỜNG THI CÔNG" ===
    "I.440": (
        "Đoạn đường đang thi công",
        "Phía trước đường đang thi công sửa chữa, giảm tốc độ và chú ý."
    ),
}

count_fixed = 0
for label, (new_meaning, new_advice) in fixes.items():
    mask = df["label"] == label
    n = mask.sum()
    if n > 0:
        old_meaning = df.loc[mask, "meaning"].iloc[0]
        df.loc[mask, "meaning"] = new_meaning
        df.loc[mask, "advice"] = new_advice
        print(f"  FIX {label} ({n} rows): '{old_meaning}' -> '{new_meaning}'")
        count_fixed += 1
    else:
        print(f"  SKIP {label}: not found in metadata")

df.to_csv("d:/gtvn/data/metadata.csv", index=False, encoding="utf-8-sig")
print(f"\nDone! Fixed {count_fixed} labels in metadata.csv")
