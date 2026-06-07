# Nhận diện Biển báo Giao thông Việt Nam

Hệ thống nhận diện biển báo giao thông đường bộ ở Việt Nam từ ảnh đầu vào. Người dùng có thể tải ảnh, dán ảnh hoặc kéo thả ảnh vào giao diện, sau đó khoanh vùng biển báo cần nhận diện. Backend xử lý ảnh, tạo vector embedding bằng CLIP, tìm kiếm ảnh mẫu gần nhất bằng FAISS và trả về nhóm biển, mã biển, ý nghĩa, lời khuyên và độ tương đồng.

Dữ liệu được tổ chức theo định hướng QCVN 41, gồm 5 nhóm biển báo giao thông: biển cấm, biển nguy hiểm, biển hiệu lệnh, biển chỉ dẫn và biển phụ.

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Hệ điều hành | Windows 10+, macOS 12+, Ubuntu 20.04+ |
| Python | 3.10 trở lên |
| Node.js | 18 trở lên |
| RAM | Tối thiểu 8 GB, khuyến nghị 16 GB |
| GPU | Không bắt buộc, có CUDA thì chạy nhanh hơn |
| Dung lượng | Khoảng vài GB do có dataset, FAISS index và model CLIP |

## 2. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| CLIP ViT-Large/14 | Chuyển ảnh biển báo thành vector embedding 768 chiều |
| FAISS IndexFlatIP | Tìm kiếm vector ảnh mẫu gần nhất |
| FastAPI | Xây dựng backend API |
| PyTorch | Chạy mô hình CLIP |
| Pillow | Đọc ảnh, chuyển RGB, resize, lật ảnh |
| Pandas / NumPy | Xử lý metadata và ma trận embedding |
| React + Vite | Xây dựng giao diện người dùng |

## 3. Dữ liệu hiện tại

| Nội dung | Giá trị |
|---|---|
| Số mã biển | 326 mã biển |
| Số dòng metadata/index | 6.983 ảnh |
| Kích thước vector | 768 chiều |
| File metadata | `data/metadata.csv` |
| File embedding | `data/image_embeddings.npy` |
| File FAISS index | `data/faiss_index.faiss` |
| Thư mục ảnh | `dataset_aug/` |
| Thư mục test | `data_test/` |

5 nhóm biển theo cấu trúc dữ liệu:

| Nhóm | Thư mục tương ứng |
|---|---|
| Biển cấm | `dataset_aug/Prohibitory Signs/` |
| Biển nguy hiểm | `dataset_aug/Warning Signs/` |
| Biển hiệu lệnh | `dataset_aug/Mandatory Signs/` |
| Biển chỉ dẫn | `dataset_aug/Information Signs/` |
| Biển phụ | `dataset_aug/Supplementary Signs/` |


## 4. Clone project từ GitHub

Mở Terminal hoặc PowerShell tại thư mục muốn lưu project, sau đó chạy:

**Windows / PowerShell**

```powershell
git clone https://github.com/DinhVen/GTVN.git
cd GTVN
```

**macOS / Linux**

```bash
git clone https://github.com/DinhVen/GTVN.git
cd GTVN
```

## 5. Cài đặt Backend

Khuyến nghị tạo môi trường ảo để thư viện của project không lẫn với máy cá nhân.

**Windows / PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r backend\requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Lần đầu chạy backend, model CLIP có thể được tải về máy nên sẽ mất thời gian hơn các lần sau.

## 6. Chạy Backend

Chạy từ thư mục gốc của project:

**Windows / PowerShell**

```powershell
python backend\main.py
```

**macOS / Linux**

```bash
python3 backend/main.py
```

Backend chạy tại:

```text
http://localhost:8000
```

API chính:

```text
POST http://localhost:8000/search
```

## 7. Cài đặt và chạy Frontend

Mở thêm một Terminal hoặc PowerShell mới, vẫn ở thư mục gốc project.

**Windows / PowerShell**

```powershell
cd frontend
npm install
npm run dev
```

**macOS / Linux**

```bash
cd frontend
npm install
npm run dev
```

Frontend chạy tại:

```text
http://localhost:5173
```

Khi sử dụng:

1. Chọn ảnh, kéo thả ảnh hoặc dán ảnh bằng `Ctrl + V`.
2. Khoanh vùng biển báo cần nhận diện.
3. Chọn dạng crop phù hợp: chữ nhật, tròn hoặc tam giác.
4. Bấm **Cắt ảnh & Nhận diện**.
5. Xem kết quả gồm độ tương đồng, nhóm biển, mã biển, ý nghĩa và lời khuyên.

## 8. Chạy lại FAISS khi thay đổi dữ liệu

Nếu chỉ clone project về để demo thì không cần chạy bước này vì project đã có sẵn `metadata.csv`, `image_embeddings.npy` và `faiss_index.faiss`.

Chỉ chạy khi có thay đổi dữ liệu trong `dataset_aug/` hoặc `data/metadata.csv`.

**Windows / PowerShell**

```powershell
python backend\rebuild_faiss.py
```

**macOS / Linux**

```bash
python3 backend/rebuild_faiss.py
```

Script này sẽ:

1. Đọc từng dòng trong `data/metadata.csv`.
2. Lấy `image_path` để mở ảnh tương ứng.
3. Bỏ qua ảnh lỗi hoặc ảnh không đọc được.
4. Dùng CLIP để encode ảnh thành vector 768 chiều.
5. Lưu lại `image_embeddings.npy`.
6. Tạo lại `faiss_index.faiss`.
7. Đồng bộ lại `metadata.csv` với các ảnh hợp lệ.

## 9. Tạo dữ liệu augment

Chỉ chạy khi muốn sinh thêm biến thể ảnh từ ảnh gốc.

**Windows / PowerShell**

```powershell
python backend\augment.py
```

**macOS / Linux**

```bash
python3 backend/augment.py
```

Sau khi augment xong, cần chạy lại:

```powershell
python backend\rebuild_faiss.py
```

Trên macOS / Linux:

```bash
python3 backend/rebuild_faiss.py
```

## 10. Đánh giá kết quả

Cần chạy backend trước, sau đó chạy script đánh giá.

**Windows / PowerShell**

```powershell
python backend\evaluate.py
```

**macOS / Linux**

```bash
python3 backend/evaluate.py
```

Script sẽ gửi từng ảnh trong `data_test/` lên API `/search`, so sánh nhãn dự đoán với nhãn thật và tạo file:

```text
evaluation_results.csv
```

Các chỉ số thường xem:

| Chỉ số | Ý nghĩa |
|---|---|
| Top-1 accuracy | Kết quả đúng nằm ở vị trí đầu tiên |
| Top-2 accuracy | Kết quả đúng nằm trong 2 kết quả đầu |
| Top-3 accuracy | Kết quả đúng nằm trong 3 kết quả đầu |

## 11. Cấu trúc thư mục

```text
GTVN/
  backend/
    main.py              # Backend FastAPI, API /search
    augment.py           # Sinh biến thể ảnh
    rebuild_faiss.py     # Tạo embedding và FAISS index
    evaluate.py          # Đánh giá Top-1, Top-2, Top-3
    requirements.txt

  frontend/
    src/
      App.jsx            # Giao diện chính
      App.css            # Style giao diện

  data/
    metadata.csv         # Thông tin ảnh, nhóm biển, mã biển, ý nghĩa, lời khuyên
    image_embeddings.npy # Ma trận vector ảnh
    faiss_index.faiss    # Chỉ mục FAISS

  dataset_aug/
    Prohibitory Signs/   # Biển cấm
    Warning Signs/       # Biển nguy hiểm
    Mandatory Signs/     # Biển hiệu lệnh
    Information Signs/   # Biển chỉ dẫn
    Supplementary Signs/ # Biển phụ

  data_test/             # Ảnh dùng để đánh giá
```

## 12. Hệ thống tổng quan nhận diện

Khi người dùng gửi ảnh lên API `/search`, backend xử lý theo các bước:

1. Kiểm tra file upload có phải ảnh không.
2. Đọc ảnh bằng Pillow và chuyển sang RGB.
3. Resize ảnh về `224x224`.
4. Encode ảnh bằng CLIP để tạo vector 768 chiều.
5. Tìm top 50 ảnh mẫu gần nhất bằng FAISS.
6. OOD check: nếu điểm top-1 thấp hơn `0.70` thì từ chối vì ảnh có thể không phải biển báo.
7. Group Voting: cộng điểm cho nhóm biển xuất hiện nhiều trong top ứng viên.
8. Mirror Fix: xử lý các cặp biển dễ nhầm trái/phải.
9. Lọc trùng theo `label`.
10. Trả JSON kết quả cho frontend.

Ví dụ JSON trả về:

```json
{
  "results": [
    {
      "rank": 1,
      "score": 0.95,
      "label": "P.123b",
      "group": "cấm",
      "meaning": "Cấm rẽ phải",
      "advice": "Không được rẽ phải, đi thẳng hoặc rẽ trái.",
      "image_path": "dataset_aug/Prohibitory Signs/P.123b/1_original.png"
    }
  ]
}
```

## 13. Ghi chú khi triển khai

- Backend phải chạy trước frontend.
- Nếu frontend không gọi được API, kiểm tra backend có đang chạy tại `localhost:8000` không.
- Nếu ảnh kết quả không hiện, kiểm tra `image_path` trong `metadata.csv` và mount static `/dataset_aug`.
- Nếu thay đổi dữ liệu, phải chạy lại `rebuild_faiss.py`.
- Không cần upload file báo cáo Word/PowerPoint lên GitHub; các file này đã được đưa vào `.gitignore`.

