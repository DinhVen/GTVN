# Nhận diện Biển Báo Giao Thông Việt Nam bằng Embeddings

Dự án sử dụng phương pháp Content-Based Image Retrieval (CBIR) để nhận diện biển báo giao thông đường bộ Việt Nam (theo QCVN 41), tận dụng sức mạnh của mô hình Vision-Language CLIP thay vì thuật toán phân loại CNN truyền thống.

## Tính năng Nổi bật & Đóng góp Chính

1. Nhận diện Không cần Huấn luyện lại (Zero-shot Embedding):
Hệ thống mã hóa ảnh biển báo thành các vector đặc trưng (768 chiều) thông qua kiến trúc mô hình openai/clip-vit-large-patch14. Bước tìm kiếm và truy vấn được thực hiện cực nhanh qua cơ sở dữ liệu lõi FAISS (Facebook AI Similarity Search) bằng phép đo Cosine. Cho phép bổ sung hàng loạt biển báo mới cực kỳ linh hoạt.

2. Thuật toán Multi-modal Re-ranking (Tái xếp hạng kép cả Hình và Chữ):
Việc dùng thuật toán này khắc phục hoàn toàn khuyết điểm "Mù phương hướng" (ví dụ: nhầm lẫn giữa mũi tên Rẽ Trái và Rẽ Phải) tồn đọng ở mô hình suy luận Zero-shot. Backend sẽ tìm Top 15 biển có hình dạng giống nhất, sau đó đem Tên ý nghĩa của 15 biển báo này gọi qua não bộ Ngôn ngữ (Text Encoder) của CLIP để chấm điểm ngược lại với Hình ảnh truy vấn ban đầu. Độ dự phóng rủi ro sai hướng bị triệt tiêu hoàn toàn nhờ sự áp chế sai số của phần Ngôn ngữ.

3. Lọc Trùng Lặp thông minh (Deduplicator):
Hệ thống tối ưu tự động gom nhóm để đảm bảo hiện đúng 3 loại biển báo khác nhau hoàn toàn trên kết quả Frontend, tránh nhiễu thông tin cho người dùng đầu cuối.

4. Xử lý nhiễu bằng Giao diện Tương tác:
Giao diện cung cấp công cụ tự động cắt cúp linh hoạt, giúp người dùng loại bỏ các mảng rừng cây/cột sáng thừa thải mà không cần phải triển khai bất kỳ mô hình Nhận diện Vật thể (Object Detection - YOLO) nào. 

## Yêu cầu Hệ thống

- Python 3.8 trở lên (Khuyến nghị 3.9 tới 3.10)
- Node.js 16+ và npm
- Git

## Hướng dẫn Cài đặt & Khởi động Dự án

Bước 1: Clone project về máy
git clone https://github.com/DinhVen/GTVN.git
cd GTVN

Bước 2: Cài đặt Backend (Python FastAPI)
1. Di chuyển vào thư mục backend:
   cd backend
2. Tạo và kích hoạt môi trường ảo:
   python -m venv venv
   - Trên Windows: venv\Scripts\activate
   - Trên Linux/Mac: source venv/bin/activate
3. Cài đặt các thư viện cần thiết:
   pip install -r requirements.txt
4. Khởi tạo Cơ sở Dữ liệu Vector lõi (FAISS Index):
   python rebuild_faiss.py
   (Lưu ý: Quá trình sẽ cần ít phút để tính toán vector 768-D cho hơn 3400 tấm ảnh mẫu)
5. Chạy backend server:
   python main.py
   (Backend chạy tại: http://localhost:8000)

Bước 3: Cài đặt Frontend (React + Vite)
1. Mở thêm 1 terminal thứ 2 và vào folder frontend:
   cd frontend
2. Cài đặt thư viện Node:
   npm install
3. Chạy giao diện Local:
   npm run dev
   (Frontend chạy tại: http://localhost:5173)

Bước 4: Trải nghiệm
Mở trình duyệt web truy cập http://localhost:5173 
Bấm nút chọn ảnh, hoặc dán ảnh (Ctrl+V) trực tiếp vào màn hình. Dùng chuột căn chỉnh khung viền crop và ấn Nhận diện.
