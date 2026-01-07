# 🌫️ Hướng Dẫn Sử Dụng Hệ Thống Dự Đoán Ô Nhiễm Không Khí Hà Nội

## 📋 Mục Lục

1. [Giới Thiệu Chung](#giới-thiệu-chung)
2. [Cài Đặt và Khởi Chạy](#cài-đặt-và-khởi-chạy)
3. [Giao Diện Chính](#giao-diện-chính)
4. [Các Chức Năng Chi Tiết](#các-chức-năng-chi-tiết)
5. [Hướng Dẫn Sử Dụng Từng Bước](#hướng-dẫn-sử-dụng-từng-bước)
6. [Câu Hỏi Thường Gặp](#câu-hỏi-thường-gặp)
7. [Lưu Ý Quan Trọng](#lưu-ý-quan-trọng)

---

## 🎯 Giới Thiệu Chung

Hệ thống Dự Đoán Ô Nhiễm Không Khí Hà Nội là một ứng dụng web tương tác sử dụng thuật toán học máy để:

- **Phân tích dữ liệu chất lượng không khí** của Hà Nội
- **Dự đoán chỉ số AQI** (Air Quality Index)
- **Phân loại mức độ ô nhiễm** theo tiêu chuẩn quốc tế
- **So sánh hiệu suất** của 4 thuật toán học máy khác nhau

### 📊 Các Thuật Toán Được Sử Dụng

| Thuật Toán | Chức Năng | Ưu Điểm |
|-----------|-----------|----------|
| **Hồi Quy Tuyến Tính** | Dự đoán AQI | Nhanh, dễ diễn giải |
| **Cây Quyết Định (CART)** | Dự đoán AQI | Xử lý dữ liệu phi tuyến |
| **SVM** | Phân loại mức độ ô nhiễm | Độ chính xác cao |
| **Hồi Quy Logistic** | Phân loại mức độ ô nhiễm | Đáng tin cậy |

---

## 🚀 Cài Đặt và Khởi Chạy

### Yêu Cầu Hệ Thống
- Python 3.8 trở lên
- 4GB RAM tối thiểu
- Trình duyệt web hiện đại

### Các Bước Cài Đặt

1. **Mở Terminal/Command Prompt**
2. **Di chuyển đến thư mục dự án:**
   ```bash
   cd C:\Users\Acer\Downloads\BLTHocMay
   ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install streamlit pandas numpy scikit-learn plotly
   ```

4. **Khởi chạy ứng dụng:**
   ```bash
   streamlit run main.py
   ```

5. **Mở trình duyệt** và truy cập `http://localhost:8501`

---

## 🖥️ Giao Diện Chính

### Thanh Điều Hướng Bên Trái
- **🔄 Tải Lại Dữ Liệu**: Làm mới toàn bộ dữ liệu
- **Menu lựa chọn trang**: Chuyển đổi giữa các chức năng

### Các Trang Chính

1. **🏠 Dashboard Chính** - Tổng quan hệ thống
2. **🔍 Tìm Kiếm Theo Thời Gian** - Phân tích dữ liệu theo khoảng thời gian
3. **🔧 Tiền Xử Lý Dữ Liệu** - Làm sạch và chuẩn bị dữ liệu
4. **🤖 Huấn Luyện Mô Hình** - Train các thuật toán ML
5. **📊 Đánh Giá & So Sánh Mô Hình** - Đánh giá hiệu suất
6. **🔮 Dự Đoán Thời Gian Thực** - Dự báo AQI tức thì
7. **📋 Kết Luận & Khuyến Nghị** - Tổng kết và tư vấn

---

## 📖 Các Chức Năng Chi Tiết

### 1. 🏠 Dashboard Chính

**Mục đích:** Cung cấp cái nhìn tổng quan về hệ thống và dữ liệu

**Các thành phần:**
- **Thông tin dữ liệu:** Tổng số mẫu, khoảng thời gian, nguồn dữ liệu
- **Thống kê AQI:** Giá trị trung bình, cao nhất, thấp nhất
- **Phân phối mức độ ô nhiễm:** Tỷ lệ các mức (Tốt, Trung Bình, Kém, Xấu, Rất Xấu, Nguy Hiểm)
- **Biểu đồ tương quan:** Mối quan hệ giữa các chất ô nhiễm

**Cách sử dụng:**
1. Trang sẽ tự động tải khi mở ứng dụng
2. Xem các thống kê tổng quan
3. Kiểm tra chất lượng dữ liệu

---

### 2. 🔍 Tìm Kiếm Theo Thời Gian

**Mục đích:** Phân tích dữ liệu trong khoảng thời gian cụ thể

**Các bước sử dụng:**

#### Bước 1: Chọn Khoảng Thời Gian
**Tùy chọn nhanh:**
- **7 Ngày Gần Đây**: Dữ liệu 7 ngày vừa qua
- **30 Ngày Gần Đây**: Dữ liệu 30 ngày vừa qua  
- **Tháng Này**: Từ đầu tháng đến nay
- **Tháng Trước**: Toàn bộ tháng trước
- **Năm Này**: Từ đầu năm đến nay
- **Tùy Chọn**: Chọn ngày cụ thể

**Tùy chọn theo giờ (tùy chọn):**
- Tick "Lọc theo giờ"
- Chọn giờ bắt đầu và kết thúc

#### Bước 2: Thực Hiện Tìm Kiếm
1. Nhấn nút **🔍 Tìm Kiếm**
2. Chờ hệ thống xử lý
3. Xem kết quả hiển thị

#### Bước 3: Xuất Dữ Liệu
**Các tùy chọn export:**
- **📥 Export CSV**: Xuất toàn bộ dữ liệu đã lọc
- **📊 Export Full CSV**: Xuất dữ liệu đầy đủ
- **📈 Export Summary**: Xuất báo cáo tóm tắt
- **📋 Export Statistics**: Xuất báo cáo thống kê chi tiết

---

### 3. 🔧 Tiền Xử Lý Dữ Liệu

**Mục đích:** Làm sạch và chuẩn bị dữ liệu cho huấn luyện mô hình

**Các bước thực hiện:**

1. **Kiểm tra dữ liệu gốc:**
   - Xem thống kê dữ liệu ban đầu
   - Kiểm tra giá trị thiếu, ngoại lệ

2. **Áp dụng tiền xử lý:**
   - Nhấn nút **🔧 Áp Dụng Tiền Xử Lý**
   - Hệ thống sẽ tự động:
     - Xử lý giá trị thiếu
     - Loại bỏ ngoại lệ
     - Chuẩn hóa dữ liệu
     - Tách features và labels

3. **Kiểm tra kết quả:**
   - Xem thống kê dữ liệu sau xử lý
   - Kiểm tra kích thước tập train/test

---

### 4. 🤖 Huấn Luyện Mô Hình

**Mục đích:** Huấn luyện các thuật toán học máy

**Các bước thực hiện:**

#### Bước 1: Chọn Mô Hình
Chọn các mô hình muốn huấn luyện:
- ✅ **Hồi Quy Tuyến Tính**
- ✅ **Cây Quyết Định (CART)**
- ✅ **SVM**
- ✅ **Hồi Quy Logistic**

#### Bước 2: Cấu Hình Tham Số (Tùy chọn)
**Hồi Quy Tuyến Tính:**
- Kiểu regularization: Ridge, Lasso, None
- Tham số alpha: 0.1 - 10.0

**Cây Quyết Định:**
- Chiều sâu tối đa: 3 - 10
- Số mẫu tối thiểu: 2 - 10
- Tiêu chí: gini, entropy

**SVM:**
- C (Độ Ngược): 0.1, 1, 10, 100
- Kernel: linear, rbf, poly
- Gamma: scale, auto
- Degree (cho polynomial): 2 - 5

**Hồi Quy Logistic:**
- C (Độ Ngược): 0.1, 1, 10, 100
- Penalty: l1, l2
- Solver: liblinear, saga
- Max Iterations: 100 - 2000

#### Bước 3: Bắt Đầu Huấn Luyện
1. Nhấn nút **🚀 Huấn Luyện Các Mô Hình Đã Chọn**
2. Theo dõi tiến trình huấn luyện
3. Xem kết quả huấn luyện

---

### 5. 📊 Đánh Giá & So Sánh Mô Hình

**Mục đích:** So sánh hiệu suất các mô hình đã huấn luyện

**Các chức năng:**

#### Bảng Tổng Kết
- So sánh tất cả mô hình
- Các chỉ số: MSE, RMSE, MAE, R², Accuracy, F1-Score
- Xác định mô hình tốt nhất

#### Biểu Đồ So Sánh
- **Biểu đồ cột**: So sánh chỉ số hồi quy
- **Biểu đồ tròn**: So sánh độ chính xác phân loại
- **Biểu đồ radar**: Tổng quan hiệu suất

#### Phân Tích Chi Tiết
- **Tầm quan trọng đặc trưng**: Các yếu tố ảnh hưởng nhất
- **Ma trận nhầm lẫn**: Chi tiết lỗi phân loại
- **Biểu đồ dự đoán vs thực tế**: Đánh giá trực quan

---

### 6. 🔮 Dự Đoán Thời Gian Thực

**Mục đích:** Dự đoán AQI với tham số người dùng nhập

**Các bước sử dụng:**

#### Bước 1: Nhập Tham Số Môi Trường

**Chất ô nhiễm (μg/m³):**
- **PM2.5**: 5 - 200 (bụi mịn)
- **PM10**: 10 - 300 (bụi thô)
- **NO₂**: 5 - 150 (Nitơ dioxit)
- **SO₂**: 2 - 100 (Lưu huỳnh dioxit)
- **CO**: 0.5 - 10 mg/m³ (Carbon monoxide)
- **O₃**: 10 - 200 (Ozone)

**Thời tiết:**
- **Nhiệt độ**: -10°C đến 50°C
- **Độ ẩm**: 30% đến 95%
- **Tốc độ gió**: 0.5 đến 10 m/s
- **Áp suất**: 900 đến 1100 hPa
- **Lượng mưa**: 0 đến 100 mm

#### Bước 2: Thực Hiện Dự Đoán
1. Nhấn nút **🔮 Dự Đoán AQI**
2. Xem kết quả từ tất cả mô hình
3. So sánh dự đoán

#### Bước 3: Xem Khuyến Nghị
- **Mức độ ô nhiễm**: Tốt, Trung Bình, Kém, Xấu, Rất Xấu, Nguy Hiểm
- **Khuyến nghị sức khỏe**: Hành động đề xuất
- **Màu sắc cảnh báo**: Dễ nhận biết

---

### 7. 📋 Kết Luận & Khuyến Nghị

**Mục đích:** Tổng kết hiệu suất và đưa ra khuyến nghị

**Nội dung:**
- **Bảng xếp hạng mô hình**: Xếp hạng theo hiệu suất
- **Khuyến nghị sử dụng**: Mô hình phù hợp cho từng trường hợp
- **Gợi ý cải tiến**: Hướng phát triển tương lai
- **Tài liệu tham khảo**: Nguồn thông tin

---

## 🔄 Hướng Dẫn Sử Dụng Từng Bước

### Quy Trình Hoàn Chỉnh (Dành cho người mới)

#### Bước 1: Khám Phá Dữ Liệu
1. Mở ứng dụng → **Dashboard Chính**
2. Xem thống kê tổng quan
3. Chuyển đến **Tìm Kiếm Theo Thời Gian**
4. Chọn "7 Ngày Gần Đây" → **Tìm Kiếm**
5. Xuất dữ liệu để kiểm tra

#### Bước 2: Chuẩn Bị Dữ Liệu
1. Chuyển đến **Tiền Xử Lý Dữ Liệu**
2. Nhấn **🔧 Áp Dụng Tiền Xử Lý**
3. Kiểm tra kết quả xử lý

#### Bước 3: Huấn Luyện Mô Hình
1. Chuyển đến **Huấn Luyện Mô Hình**
2. Chọn tất cả 4 mô hình
3. Sử dụng tham số mặc định
4. Nhấn **🚀 Huấn Luyện Các Mô Hình Đã Chọn**

#### Bước 4: Đánh Giá
1. Chuyển đến **Đánh Giá & So Sánh Mô Hình**
2. Xem bảng tổng kết
3. Phân tích biểu đồ
4. Xác định mô hình tốt nhất

#### Bước 5: Dự Đoán
1. Chuyển đến **Dự Đoán Thời Gian Thực**
2. Nhập các tham số môi trường
3. Nhấn **🔮 Dự Đoán AQI**
4. Xem kết quả và khuyến nghị

#### Bước 6: Tổng Kết
1. Chuyển đến **Kết Luận & Khuyến Nghị**
2. Đọc các khuyến nghị
3. Lưu lại kết quả quan trọng

---

## ❓ Câu Hỏi Thường Gặp

### Q1: Tại sao ứng dụng không tải được dữ liệu?
**A:** Kiểm tra:
- File `hanoi_air_quality_recent.csv` có tồn tại không
- Kết nối mạng có ổn định không
- Thử nhấn **🔄 Tải Lại Dữ Liệu**

### Q2: Huấn luyện mô hình mất nhiều thời gian?
**A:** Thời gian huấn luyện phụ thuộc vào:
- Số lượng dữ liệu
- Số mô hình đã chọn
- Độ phức tạp của tham số
- **Giải pháp:** Chọn ít mô hình hơn hoặc dùng tham số mặc định

### Q3: Kết quả dự đoán không chính xác?
**A:** Có thể do:
- Dữ liệu huấn luyện không đủ
- Tham số môi trường ngoài khoảng thực tế
- **Giải pháp:** Sử dụng nhiều dữ liệu hơn, kiểm tra lại tham số

### Q4: Làm thế nào để xuất dữ liệu?
**A:** Trong trang **Tìm Kiếm Theo Thời Gian**:
1. Chọn khoảng thời gian
2. Nhấn **Tìm Kiếm**
3. Chọn loại export mong muốn
4. Nhấn nút download

### Q5: Mô hình nào là tốt nhất?
**A:** Tùy vào mục đích:
- **Độ chính xác cao nhất**: SVM
- **Nhanh nhất**: Hồi Quy Tuyến Tính
- **Dễ diễn giải**: Cây Quyết Định
- **Ổn định nhất**: Hồi Quy Logistic

---

## ⚠️ Lưu Ý Quan Trọng

### 🔒 Bảo Mật Dữ Liệu
- Dữ liệu chỉ dùng cho mục đích nghiên cứu
- Không chứa thông tin cá nhân
- Lưu trữ cục bộ trên máy

### 🎯 Hạn Chế
- Dữ liệu giả lập, không phải dữ liệu thực tế
- Mô hình cần được cập nhật định kỳ
- Kết quả dự đoán chỉ mang tính tham khảo

### 💡 Mẹo Sử Dụng
1. **Luôn tải lại dữ liệu** khi bắt đầu phiên mới
2. **Lưu kết quả quan trọng** trước khi đóng ứng dụng
3. **Sử dụng tham số thực tế** khi dự đoán
4. **Kiểm tra chất lượng dữ liệu** trước khi huấn luyện
5. **So sánh nhiều mô hình** để có kết quả tốt nhất

### 🐨 Xử Lý Lỗi
- **Lỗi dữ liệu**: Kiểm tra file CSV, tải lại dữ liệu
- **Lỗi huấn luyện**: Giảm số mô hình, đơn giản hóa tham số
- **Lỗi dự đoán**: Kiểm tra lại tham số nhập vào
- **Lỗi export**: Kiểm tra kết nối, thử lại sau

---

## 📞 Hỗ Trợ Kỹ Thuật

### Khi Gặp Vấn Đề:
1. **Kiểm tra console** của trình duyệt
2. **Xem terminal output** để biết lỗi chi tiết
3. **Tải lại trang** và thử lại
4. **Sử dụng dữ liệu mặc định** nếu có lỗi file

### Liên Hệ:
- Kiểm tra file README_VI.md để biết thông tin dự án
- Xem code comments để hiểu chi tiết
- Sử dụng các công cụ debug tích hợp

---

## 🎉 Chúc Mừng!

Bạn đã hoàn thành hướng dẫn sử dụng! Bây giờ bạn có thể:

✅ Hiểu rõ các chức năng của ứng dụng  
✅ Sử dụng thành thạo tất cả tính năng  
✅ Phân tích và dự đoán chất lượng không khí  
✅ Đưa ra quyết định dựa trên dữ liệu khoa học  

**Chúc bạn có trải nghiệm hữu ích với hệ thống dự đoán ô nhiễm không khí Hà Nội!** 🌫️✨

---

*Phiên bản tài liệu: 1.0 | Cập nhật: 07/01/2026*