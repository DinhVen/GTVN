# Nhận diện Biển báo Giao thông Việt Nam

Hệ thống nhận diện biển báo giao thông Việt Nam bằng phương pháp **Embedding-based Image Retrieval**, sử dụng mô hình **CLIP** để trích xuất đặc trưng ảnh và **FAISS** để tìm kiếm tương đồng. Dữ liệu tuân theo tiêu chuẩn **QCVN 41:2024/BGTVT**.

## Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| **OS** | Windows 10+, Ubuntu 20.04+, macOS 12+ |
| **Python** | 3.10 trở lên |
| **Node.js** | 18 trở lên |
| **RAM** | 8 GB (khuyến nghị 16 GB) |
| **GPU** | Không bắt buộc (hỗ trợ CUDA nếu có) |
| **Dung lượng** | ~5 GB (model CLIP + dataset + FAISS index) |

## Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| **CLIP** (ViT-Large/14) | Trích xuất vector đặc trưng 768 chiều từ ảnh |
| **FAISS** (IndexFlatIP) | Tìm kiếm vector tương đồng (Inner Product) |
| **FastAPI** | Backend API server |
| **PyTorch** | Framework deep learning, chạy inference CLIP |
| **React + Vite** | Frontend giao diện người dùng |
| **Pillow / OpenCV** | Xử lý ảnh, data augmentation |
| **Pandas** | Quản lý metadata |

## Dữ liệu

| Thông số | Giá trị |
|---|---|
| Tổng số biển | 326 biển (4 nhóm) |
| Tổng ảnh (sau augment) | ~7,058 ảnh |
| Nhóm Cấm | Prohibitory Signs |
| Nhóm Nguy hiểm | Warning Signs |
| Nhóm Hiệu lệnh | Mandatory Signs |
| Nhóm Chỉ dẫn | Information Signs |
| Augmentation | 18 biến thể/biển (xoay, mờ, sáng/tối, phối cảnh, zoom) |
| Vector dimension | 768 chiều |
| Tập test | ~300 ảnh riêng biệt |


## Triển khai hệ thống

### 1. Cài đặt Backend

**Windows (PowerShell):**
```powershell
cd backend
pip install -r requirements.txt
```

**Linux / macOS (Terminal):**
```bash
cd backend
pip3 install -r requirements.txt
```

> Nếu dùng GPU (CUDA), cài thêm: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### 2. Chạy Backend

**Windows:**
```powershell
python backend\main.py
```

**Linux / macOS:**
```bash
python3 backend/main.py
```

Server chạy tại `http://localhost:8000`.

### 3. Chạy Frontend

**Windows:**
```powershell
cd frontend
npm install
npm run dev
```

**Linux / macOS:**
```bash
cd frontend
npm install
npm run dev
```

Mở `http://localhost:5173` để sử dụng.

### 4. Cập nhật dataset (khi thêm/xóa biển)

**Windows:**
```powershell
python backend\augment.py
python backend\rebuild_faiss.py
```

**Linux / macOS:**
```bash
python3 backend/augment.py
python3 backend/rebuild_faiss.py
```

### 5. Đánh giá accuracy (cần server đang chạy)

**Windows:**
```powershell
python backend\evaluate.py
```

**Linux / macOS:**
```bash
python3 backend/evaluate.py
```

Kết quả lưu tại `evaluation_results.csv`.

## Cấu trúc

```
backend/
  main.py              — API server (pipeline 5 bước)
  augment.py           — Data augmentation (18 biến thể/biển)
  rebuild_faiss.py     — Tạo FAISS index
  evaluate.py          — Đánh giá accuracy
  requirements.txt
frontend/src/
  App.jsx              — Giao diện (upload, crop, kết quả)
  App.css
data/
  metadata.csv         — Metadata toàn bộ ảnh
  faiss_index.faiss    — Vector index
dataset_aug/           — ~326 biển × ~19 ảnh/biển
data_test/             — Tập test
```

## Pipeline nhận diện

1. **Resize** — 224×224
2. **CLIP encode** — vector 768 chiều
3. **FAISS search + OOD check** — top 50, reject nếu score < 0.45
4. **Group Voting** — cộng điểm nhóm đa số
5. **Mirror fix** — sửa nhầm trái/phải

## Công nghệ

- **Backend:** FastAPI, PyTorch, CLIP ViT-Large/14, FAISS
- **Frontend:** React, Vite
