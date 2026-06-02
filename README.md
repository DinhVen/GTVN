# Nhận diện biển báo giao thông Việt Nam bằng CLIP Embeddings và FAISS

Dự án xây dựng hệ thống nhận diện biển báo giao thông đường bộ ở Việt Nam từ ảnh đầu vào. Hệ thống tiếp cận bài toán theo hướng tìm kiếm ảnh tương đồng dựa trên vector embedding, thay vì huấn luyện một mô hình phân loại ảnh từ đầu.

Pipeline chính:

```text
Ảnh người dùng -> FastAPI Backend -> CLIP Embedding -> OOD Check
               -> FAISS Search -> Text Re-ranking -> Mirror Fix
               -> JSON Response -> ReactJS Frontend
```

## 1. Công nghệ sử dụng

- **CLIP**: dùng model `openai/clip-vit-large-patch14` để encode ảnh và text thành vector embedding 768 chiều.
- **FAISS**: tìm kiếm vector ảnh mẫu gần nhất với vector ảnh đầu vào.
- **FastAPI**: xây dựng backend API, endpoint chính là `/search`.
- **ReactJS + Vite**: xây dựng giao diện upload/crop ảnh và hiển thị kết quả.
- **PIL / Pillow**: đọc ảnh, chuyển RGB, resize ảnh.
- **PyTorch**: chạy mô hình CLIP và tính toán tensor.
- **Pandas / NumPy**: xử lý metadata và embedding.

## 2. Cấu trúc thư mục chính

```text
GTVN/
├── backend/
│   ├── main.py              # FastAPI backend
│   ├── rebuild_faiss.py     # Tạo image_embeddings.npy và faiss_index.faiss
│   ├── evaluate.py          # Đánh giá Top-1, Top-2, Top-3 accuracy
│   └── requirements.txt     # Thư viện Python
├── data/
│   ├── metadata.csv         # Metadata của ảnh biển báo
│   ├── image_embeddings.npy # Vector ảnh mẫu
│   └── faiss_index.faiss    # Chỉ mục FAISS
├── dataset_aug/             # Tập ảnh mẫu sau tăng cường dữ liệu
├── data_test/               # Tập ảnh test để đánh giá
└── frontend/
    ├── src/
    ├── package.json
    └── vite.config.js
```

## 3. Dữ liệu

Dữ liệu được xây dựng dựa trên hệ thống biển báo giao thông đường bộ Việt Nam theo **QCVN 41**. QCVN 41 được dùng làm căn cứ để xác định mã biển, nhóm biển, ý nghĩa và chức năng của từng biển báo.

Thư mục `dataset_aug` gồm 5 nhóm biển:

```text
dataset_aug/
├── Information Signs
├── Mandatory Signs
├── Prohibitory Signs
├── Supplementary Signs
└── Warning Signs
```

File `data/metadata.csv` quản lý thông tin của từng ảnh, gồm các trường chính:

```text
image_path, group, group_source, label, meaning, class_id, advice
```

## 4. Yêu cầu hệ thống

### Backend

- Python 3.10 trở lên khuyến nghị.
- Có thể chạy bằng CPU, nhưng lần đầu load CLIP sẽ hơi chậm.
- Nếu có GPU CUDA, PyTorch có thể tận dụng GPU để xử lý nhanh hơn.

### Frontend

- Node.js 18 trở lên khuyến nghị.
- npm đi kèm Node.js.

## 5. Triển khai project

### Bước 1: Clone project

Windows và macOS / MacBook dùng chung lệnh:

```bash
git clone https://github.com/DinhVen/GTVN.git
cd GTVN
```

### Bước 2: Cài backend

Windows:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / MacBook:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Bước 3: Tạo lại FAISS index nếu cần

Nếu đã có sẵn `data/faiss_index.faiss` và `data/image_embeddings.npy` thì có thể bỏ qua bước này.

Windows:

```bash
python rebuild_faiss.py
```

macOS / MacBook:

```bash
python3 rebuild_faiss.py
```

### Bước 4: Chạy backend

Windows:

```bash
python main.py
```

macOS / MacBook:

```bash
python3 main.py
```

Backend chạy tại:

```text
http://localhost:8000
```

API nhận diện:

```text
POST http://localhost:8000/search
```

### Bước 5: Cài và chạy frontend

Mở terminal mới tại thư mục gốc project.

Windows:

```bash
cd frontend
npm install
npm run dev
```

macOS / MacBook:

```bash
cd frontend
npm install
npm run dev
```

Frontend chạy tại:

```text
http://localhost:5173
```

## 6. Cách sử dụng hệ thống

1. Mở trình duyệt và truy cập:

```text
http://localhost:5173
```

2. Chọn ảnh hoặc dán ảnh vào giao diện.
3. Crop vùng có chứa biển báo.
4. Bấm **Cắt ảnh & Nhận diện**.
5. Hệ thống hiển thị các kết quả tương đồng nhất, gồm:
   - Ảnh mẫu.
   - Độ tương đồng.
   - Loại biển.
   - Mã biển.
   - Ý nghĩa.
   - Lời khuyên.

## 7. Chạy đánh giá

Trước khi đánh giá, cần chạy backend ở port `8000`.

Windows:

```bash
cd backend
venv\Scripts\activate
python evaluate.py
```

macOS / MacBook:

```bash
cd backend
source venv/bin/activate
python3 evaluate.py
```

Kết quả chi tiết được lưu tại:

```text
evaluation_results.csv
```

Kết quả hiện tại trên tập test 342 ảnh:

```text
Top-1 Accuracy: 76.0%
Top-2 Accuracy: 81.6%
Top-3 Accuracy: 83.3%
```

## 8. Ghi chú khi triển khai

- Lần đầu chạy backend, model CLIP sẽ được tải và load vào RAM nên có thể mất thời gian.
- Nếu thay đổi dữ liệu trong `dataset_aug` hoặc `metadata.csv`, cần chạy lại:

```bash
python rebuild_faiss.py
```

hoặc trên macOS:

```bash
python3 rebuild_faiss.py
```

- Nếu frontend không gọi được backend, kiểm tra backend có chạy ở `http://localhost:8000` hay không.
- Nếu ảnh mẫu không hiển thị, kiểm tra đường dẫn `image_path` trong `metadata.csv` và thư mục `dataset_aug`.

## 9. Tài liệu tham khảo

- QCVN 41:2019/BGTVT - Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ.
- QCVN 41:2024/BGTVT - Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ.
- CLIP: https://github.com/openai/CLIP
- FAISS: https://faiss.ai/
- FastAPI: https://fastapi.tiangolo.com/
- ReactJS: https://react.dev/
