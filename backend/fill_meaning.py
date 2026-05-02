"""
Script tự động bổ sung meaning (ý nghĩa) cho các biển báo đang bị trống trong metadata.csv
Dựa theo Quy chuẩn QCVN 41:2019/BGTVT - Biển báo giao thông đường bộ Việt Nam.
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

# Từ điển ý nghĩa biển báo theo QCVN 41
MEANING_MAP = {
    # === BIỂN PHỤ (DP) ===
    "DP.127a": "Biển phụ: Khoảng cách đến đoạn đường cấm",
    "DP.127b": "Biển phụ: Khoảng cách đến đoạn đường cấm",
    "DP.127c": "Biển phụ: Khoảng cách đến đoạn đường cấm",
    "DP.133": "Biển phụ: Phương tiện bị cấm",
    "DP.134": "Biển phụ: Phạm vi tác dụng của biển",

    # === BIỂN CHỈ DẪN (I) ===
    "I.405b": "Đường cụt phía bên phải",
    "I.405c": "Đường cụt phía bên trái",
    "I.406": "Đường ưu tiên",
    "I.407b": "Đường người đi bộ sang ngang bên phải",
    "I.407c": "Đường người đi bộ sang ngang bên trái",
    "I.408a": "Đường dành cho xe đạp",
    "I.413b": "Nơi đỗ xe dành cho xe khách bên phải",
    "I.413c": "Nơi đỗ xe dành cho xe khách bên trái",
    "I.414c": "Bến xe buýt bên phải",
    "I.414d": "Bến xe buýt bên trái",
    "I.415": "Nơi đỗ xe taxi",
    "I.416": "Phà",
    "I.417b": "Bến tàu hỏa lối rẽ phải",
    "I.417c": "Bến tàu hỏa lối rẽ trái",
    "I.419a": "Bệnh viện phía trước",
    "I.419b": "Bệnh viện rẽ phải",
    "I.422a": "Trạm cấp cứu phía trước",
    "I.422b": "Trạm cấp cứu rẽ phải",
    "I.423b": "Trạm xăng dầu rẽ phải",
    "I.423c": "Trạm xăng dầu rẽ trái",
    "I.424b": "Khách sạn, nhà nghỉ rẽ phải",
    "I.424c": "Khách sạn, nhà nghỉ rẽ trái",
    "I.424d": "Nhà hàng, quán ăn bên phải",
    "I.427b": "Trạm sửa chữa rẽ phải",
    "I.433b": "Chỉ dẫn làn đường 2 làn cùng chiều",
    "I.433c": "Chỉ dẫn làn đường 3 làn cùng chiều",
    "I.433d": "Chỉ dẫn làn đường 4 làn cùng chiều",
    "I.433e": "Chỉ dẫn làn đường trên cao tốc",
    "I.434b": "Hết làn xe hạn chế",
    "I.435": "Cầu vượt liên thông",
    "I.436": "Đường trên cao",
    "I.437": "Hầm chui dân sinh",
    "I.441a": "Chỉ dẫn tên đường phố phía trước",
    "I.441b": "Chỉ dẫn tên đường phố rẽ",
    "I.441c": "Chỉ dẫn tên đường phố ngã tư",
    "I.442": "Chỉ dẫn tên cầu, sông",
    "I.443": "Chỉ dẫn số kilomet",
    "I.444a": "Chỉ dẫn địa giới hành chính tỉnh, thành phố",
    "I.444b": "Chỉ dẫn địa giới hành chính quận, huyện",
    "I.444c": "Chỉ dẫn địa giới hành chính xã, phường",
    "I.444d": "Chỉ dẫn khu vực đô thị",
    "I.444e": "Chỉ dẫn khu vực nông thôn",
    "I.444f": "Chỉ dẫn địa danh du lịch",
    "I.444g": "Chỉ dẫn địa danh lịch sử",
    "I.444h": "Chỉ dẫn địa danh văn hóa",
    "I.444i": "Chỉ dẫn khu công nghiệp",
    "I.444j": "Chỉ dẫn khu kinh tế",
    "I.444k": "Chỉ dẫn cửa khẩu quốc tế",
    "I.444l": "Chỉ dẫn sân bay",
    "I.444m": "Chỉ dẫn cảng biển",
    "I.445b": "Chỉ dẫn tuyến đường rẽ phải đi địa phương",
    "I.445c": "Chỉ dẫn tuyến đường rẽ trái đi địa phương",
    "I.445d": "Chỉ dẫn khoảng cách điểm đến rẽ phải",
    "I.445e": "Chỉ dẫn khoảng cách điểm đến rẽ trái",
    "I.445f": "Chỉ dẫn khoảng cách kiểu thẳng và rẽ",
    "I.445g": "Chỉ dẫn khoảng cách kiểu rẽ trái và phải",
    "I.445h": "Chỉ dẫn khoảng cách kiểu phức hợp",
    "I.446": "Sơ đồ chỉ dẫn đường đi trước ngã tư",
    "I.447a": "Sơ đồ chỉ dẫn tuyến đường liên tỉnh",
    "I.447b": "Sơ đồ chỉ dẫn tuyến đường nội tỉnh",
    "I.447c": "Sơ đồ chỉ dẫn tuyến đường vòng tránh",
    "I.447d": "Sơ đồ chỉ dẫn tuyến đường cao tốc",
    "I.448": "Biển chỉ dẫn số hiệu quốc lộ",
    "I.449": "Biển chỉ dẫn số hiệu đường tỉnh",

    # === BIỂN CẤM (P) ===
    "P.127a": "Cấm xe container và xe tải hạng nặng",
    "P.127b": "Cấm xe container rẽ trái",
    "P.127c": "Cấm xe container rẽ phải",
    "P.136": "Cấm sử dụng còi",
    "P.138": "Hết cấm vượt",
    "P.139": "Hết hạn chế tốc độ tối đa",
    "P.140": "Hết tất cả các lệnh cấm",

    # === BIỂN HIỆU LỆNH (R) ===
    "R.122": "Dừng lại",
    "R.302c": "Hướng đi phải theo vòng xuyến",
    "R.307": "Hướng đi vòng chướng ngại vật sang phải",
    "R.308a": "Hướng đi vòng chướng ngại vật sang trái",
    "R.308b": "Hướng đi vòng chướng ngại vật sang phải",
    "R.309": "Các xe chỉ được rẽ trái",
    "R.310a": "Các xe chỉ được rẽ phải",
    "R.310b": "Các xe chỉ được rẽ phải phía trước",
    "R.310c": "Các xe chỉ được rẽ trái phía trước",
    "R.403a": "Hướng đi trên mỗi làn đường rẽ trái và thẳng",
    "R.403b": "Hướng đi trên mỗi làn đường thẳng và rẽ phải",
    "R.403c": "Hướng đi trên mỗi làn đường rẽ trái, thẳng và rẽ phải",
    "R.403d": "Hướng đi trên mỗi làn đường rẽ trái và rẽ phải",
    "R.403e": "Hướng đi trên mỗi làn đường 2 làn rẽ trái và thẳng",
    "R.403f": "Hướng đi trên mỗi làn đường thẳng và 2 làn rẽ phải",
    "R.403g": "Hướng đi trên mỗi làn đường 4 làn phức hợp",
    "R.403h": "Hướng đi trên mỗi làn đường 3 làn phức hợp",
    "R.403k": "Hướng đi trên mỗi làn đường 5 làn phức hợp",
    "R.404a": "Hướng đi trên mỗi làn xe buýt rẽ trái",
    "R.404b": "Hướng đi trên mỗi làn xe buýt thẳng",
    "R.404c": "Hướng đi trên mỗi làn xe buýt rẽ phải",
    "R.404d": "Hướng đi trên mỗi làn xe buýt rẽ trái và thẳng",
    "R.404e": "Hướng đi trên mỗi làn xe buýt thẳng và rẽ phải",
    "R.404f": "Hướng đi trên mỗi làn xe buýt rẽ trái, thẳng, rẽ phải",
    "R.404g": "Hướng đi trên mỗi làn xe buýt rẽ trái và rẽ phải",
    "R.404h": "Hướng đi trên mỗi làn xe buýt phức hợp",
    "R.404k": "Hướng đi trên mỗi làn xe buýt phức hợp nâng cao",
    "R.411": "Tốc độ tối thiểu cho phép",
    "R.415a": "Hết hạn chế tốc độ tối thiểu",
    "R.415b": "Hết yêu cầu sử dụng xích lốp xe",
    "R.E,10a": "Phần đường dành cho xe hai bánh rẽ trái qua nút giao",
    "R.E,10b": "Phần đường dành cho xe hai bánh rẽ phải qua nút giao",
    "R.E,10c": "Phần đường dành cho xe hai bánh đi thẳng qua nút giao",
    "R.E,10d": "Phần đường dành cho xe hai bánh phức hợp qua nút giao",
    "R.E,11a": "Vạch dừng xe cho xe hai bánh bên trái",
    "R.E,11b": "Vạch dừng xe cho xe hai bánh bên phải",
    "R.E,9a": "Nơi dừng xe buýt bên phải phía trước",
    "R.E,9b": "Nơi dừng xe buýt bên trái phía trước",
    "R.E,9c": "Nơi dừng xe buýt bên phải phía sau",
    "R.E,9d": "Nơi dừng xe buýt bên trái phía sau",

    # === BIỂN NGUY HIỂM VÀ CẢNH BÁO (W) ===
    "W.207d": "Giao nhau với đường không ưu tiên bên phải",
    "W.207e": "Giao nhau với đường không ưu tiên bên trái",
    "W.207f": "Giao nhau với đường không ưu tiên chữ T ngược",
    "W.207g": "Giao nhau với đường không ưu tiên chữ T phải",
    "W.207h": "Giao nhau với đường không ưu tiên chữ T trái",
    "W.207i": "Giao nhau với đường không ưu tiên chữ Y",
    "W.207j": "Giao nhau với đường không ưu tiên chữ Y phải",
    "W.207k": "Giao nhau với đường không ưu tiên chữ Y trái",
    "W.207l": "Giao nhau với đường không ưu tiên kiểu vòng xuyến",
    "W.207m": "Giao nhau với đường không ưu tiên kiểu phức hợp",
    "W.215b": "Đường ngoặt phải và trái liên tục",
    "W.215c": "Đường ngoặt trái và phải liên tục",
    "W.216a": "Đường có ổ gà lồi lõm bên phải",
    "W.216b": "Đường có ổ gà lồi lõm bên trái",
    "W.217": "Đường hẹp cả hai bên",
    "W.218": "Đường có vách đá nguy hiểm bên phải",
    "W.222b": "Kè, vực sâu bên trái đường",
    "W.223a": "Đường trơn trượt khi trời mưa",
    "W.223b": "Đường trơn trượt khi trời mưa (dạng 2)",
    "W.228a": "Đoạn đường thường xảy ra tai nạn hướng thẳng",
    "W.228b": "Đoạn đường thường xảy ra tai nạn hướng cong phải",
    "W.228c": "Đoạn đường thường xảy ra tai nạn hướng cong trái",
    "W.228d": "Đoạn đường thường hay ùn tắc",
    "W.229": "Cảnh báo nguy hiểm khác",
    "W.230": "Đường đôi (đường có dải phân cách)",
    "W.231": "Hết đường đôi (hết dải phân cách)",
    "W.233": "Cầu có tải trọng hạn chế",
    "W.234": "Đường ngầm nước chảy, ngập nước",
    "W.235": "Đường lầy, lún",
    "W.236": "Bến phà, cầu phao",
    "W.237": "Cầu quay, cầu cất",
    "W.238": "Lối ra bờ sông, bến tàu",
    "W.239a": "Đường sắt giao nhau có rào chắn (chiều phải)",
    "W.239b": "Đường sắt giao nhau có rào chắn (chiều trái)",
    "W.240": "Cảnh báo đường sắt giao nhau không rào chắn",
    "W.241": "Đường sắt giao nhau khoảng cách 100m",
    "W.242a": "Đường sắt giao nhau khoảng cách 50m",
    "W.242b": "Đường sắt giao nhau khoảng cách gần",
    "W.243a": "Đoạn đường ngang có đường sắt (1 đường ray)",
    "W.243b": "Đoạn đường ngang có đường sắt (2 đường ray trở lên)",
    "W.243c": "Đoạn đường ngang có đường sắt (nhiều ray phức hợp)",
    "W.244": "Cảnh báo luồng gió ngược chiều mạnh",
}

def main():
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded: {len(df)} rows")

    updated_count = 0
    still_missing = []

    for idx, row in df.iterrows():
        label = str(row.get('label', ''))
        meaning = row.get('meaning', '')

        # Check if meaning is empty/NaN
        if pd.isna(meaning) or str(meaning).strip() == '':
            if label in MEANING_MAP:
                df.at[idx, 'meaning'] = MEANING_MAP[label]
                updated_count += 1
            else:
                if label not in still_missing:
                    still_missing.append(label)

    df.to_csv(METADATA_PATH, index=False)
    print(f"Updated {updated_count} rows with new meanings.")
    
    # Verify results
    empty_after = df['meaning'].isna().sum() + (df['meaning'] == '').sum()
    print(f"Remaining empty meanings: {empty_after}")
    
    if still_missing:
        print(f"\nStill missing ({len(still_missing)} labels):")
        for lbl in sorted(still_missing):
            print(f"  - {lbl}")
    else:
        print("\nAll labels now have meanings!")

if __name__ == "__main__":
    main()
