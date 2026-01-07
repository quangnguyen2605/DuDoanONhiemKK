# 🌫️ Hệ Thống Dự Đoán Ô Nhiễm Không Khí Hà Nội

## 📋 Tổng Quan Dự Án

Ứng dụng demo toàn diện này triển khai các thuật toán học máy để dự đoán chỉ số chất lượng không khí (AQI) và phân loại mức độ ô nhiễm tại Hà Nội, Việt Nam. Hệ thống so sánh 4 thuật toán khác nhau để xác định phương pháp tối ưu cho việc dự báo ô nhiễm không khí.

### 🎯 Mục Tiêu

- **Nhiệm vụ Hồi Quy**: Dự đoán giá trị AQI liên tục sử dụng các tham số môi trường
- **Nhiệm vụ Phân Loại**: Phân loại mức độ ô nhiễm thành các danh mục (Tốt, Trung Bình, Kém, Xấu, Rất Xấu, Nguy Hiểm)
- **So Sánh Thuật Toán**: Đánh giá Hồi Quy Tuyến Tính, Cây Quyết Định (CART), SVM, và Hồi Quy Logistic
- **Demo Tương Tác**: Cung cấp giao diện người dùng thân thiện cho dự đoán thời gian thực

### 🏗️ Cấu Trúc Dự Án

```
BLTHocMay/
├── main.py                    # Ứng dụng Streamlit chính
├── data_generator.py          # Tạo dữ liệu AQI Hà Nội giả lập
├── data_preprocessing.py      # Pipeline tiền xử lý dữ liệu
├── models.py                  # Triển khai mô hình học máy
├── evaluation.py              # Đánh giá và so sánh mô hình
├── visualization.py          # Công cụ trực quan hóa dữ liệu
├── requirements.txt           # Dependencies Python
├── run_app.py                 # Script khởi chạy ứng dụng
└── README_VI.md              # Tài liệu dự án (tiếng Việt)
```

## 🚀 Cài Đặt & Thiết Lập

### Yêu Cầu

- Python 3.8 trở lên
- Trình quản lý gói pip

### Các Bước Cài Đặt

1. **Tải hoặc sao chép các file dự án**
2. **Cài đặt dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy ứng dụng**:
   ```bash
   streamlit run main.py
   ```
   hoặc sử dụng script khởi chạy:
   ```bash
   python run_app.py
   ```

Ứng dụng sẽ mở trong trình duyệt web của bạn tại `http://localhost:8501`

## 📊 Thông Tin Dữ Liệu

### Bộ Dữ Liệu AQI Hà Nội Giả Lập (2024-2025)

Hệ thống tạo dữ liệu ô nhiễm không khí thực tế cho Hà Nội với các đặc điểm sau:

#### **Chất Ô Nhiễm Đo Lường**:
- **PM2.5** (μg/m³) - Bụi mịn
- **PM10** (μg/m³) - Bụi thô  
- **NO₂** (μg/m³) - Nitơ dioxit
- **SO₂** (μg/m³) - Lưu huỳnh dioxit
- **CO** (mg/m³) - Carbon monoxide
- **O₃** (μg/m³) - Ozone

#### **Yếu Tố Khí Tượng**:
- Nhiệt độ (°C)
- Độ ẩm (%)
- Tốc độ gió (m/s)
- Áp suất khí quyển (hPa)
- Lượng mưa (mm)

#### **Biến Mục Tiêu**:
- **AQI** (Chỉ số chất lượng không khí) - Giá trị liên tục (0-500)
- **Pollution_Level** - Phân loại danh mục:
  - Tốt (0-50)
  - Trung Bình (51-100)
  - Kém (101-150)
  - Xấu (151-200)
  - Rất Xấu (201-300)
  - Nguy Hiểm (301+)

### Đặc Điểm Dữ Liệu

- **Mẫu Thời Gian**: Biến đổi theo mùa, chu kỳ hàng giờ
- **Tương Quan Thực Tế**: Các chất ô nhiễm tương tác với điều kiện thời tiết
- **Giá Trị Thiếu**: 2% dữ liệu thiếu để thực tế hơn
- **Ngoại Lệ**: 1% giá trị cực đoan để kiểm tra độ robust

## 🤖 Thuật Toán Học Máy

### 1. **Hồi Quy Tuyến Tính** (Mạnh)
- **Mục Đích**: Dự đoán giá trị AQI
- **Điểm Mạnh**: Đơn giản, dễ diễn giải, huấn luyện nhanh
- **Nền Tảng Toán Học**: $y = \beta_0 + \beta_1x_1 + ... + \beta_nx_n + \epsilon$

### 2. **Cây Quyết Định (CART)** (Quang)
- **Mục Đích**: Dự đoán AQI và phân tích tầm quan trọng đặc trưng
- **Điểm Mạnh**: Mối quan hệ phi tuyến, dễ trực quan hóa
- **Thuật Toán**: Cây phân loại và hồi quy

### 3. **Support Vector Machine (SVM)** (Tiến)
- **Mục Đích**: Phân loại mức độ ô nhiễm
- **Điểm Mạnh**: Độ chính xác cao, hiệu quả trong không gian chiều cao
- **Kernels**: Linear, RBF, Polynomial

### 4. **Hồi Quy Logistic** (Thương)
- **Mục Đích**: Phân loại mức độ ô nhiễm
- **Điểm Mạnh**: Đầu ra xác suất, dự đoán nhanh
- **Nền Tảng Toán Học**: $P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + ... + \beta_nx_n)}}$

## 📈 Chỉ Số Đánh Giá

### Chỉ Số Hồi Quy
- **MSE** (Mean Squared Error): $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **RMSE** (Root Mean Squared Error): $\sqrt{MSE}$
- **MAE** (Mean Absolute Error): $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **R²** (Hệ số xác định): $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

### Chỉ Số Phân Loại
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$

## 🎨 Tính Năng Ứng Dụng

### 1. **Tạo Dữ Liệu & Khám Phá**
- Tự động tạo bộ dữ liệu giả lập
- Trực quan hóa dữ liệu tương tác
- Phân tích thống kê và ma trận tương quan
- Phân tích chuỗi thời gian

### 2. **Tiền Xử Lý Dữ Liệu**
- Xử lý giá trị thiếu
- Phát hiện và loại bỏ ngoại lệ
- Kỹ thuật đặc trưng
- Chuẩn hóa và mã hóa dữ liệu

### 3. **Huấn Luyện Mô Hình**
- Huấn luyện đa thuật toán
- Tinh chỉnh siêu tham số (Grid Search)
- Cross-validation
- So sánh hiệu suất

### 4. **Đánh Giá Mô Hình**
- Phân tích chỉ số toàn diện
- Trực quan hóa so sánh hiệu suất
- Phân tích tầm quan trọng đặc trưng
- Khuyến nghị mô hình tốt nhất

### 5. **Dự Đoán Thời Gian Thực**
- Nhập tham số tương tác
- Dự đoán AQI tức thì
- Phân loại mức độ ô nhiễm
- Khuyến nghị sức khỏe

### 6. **Kết Luận & Khuyến Nghị**
- Tóm tắt hiệu suất thuật toán
- Khuyến nghị trường hợp sử dụng
- Gợi ý cải tiến tương lai

## 🔧 Hướng Dẫn Sử Dụng

### Quy Trình Sử Dụng Từng Bước

1. **Khởi Chạy Ứng Dụng**
   ```bash
   streamlit run main.py
   ```

2. **Tạo Dữ Liệu**
   - Điều hướng đến "Tạo Dữ Liệu & Khám Phá"
   - Xem thống kê dữ liệu và trực quan hóa

3. **Tiền Xử Lý Dữ Liệu**
   - Đi đến phần "Tiền Xử Lý Dữ Liệu"
   - Nhấp "Áp Dụng Tiền Xử Lý" để làm sạch dữ liệu

4. **Huấn Luyện Mô Hình**
   - Chọn thuật toán mong muốn trong "Huấn Luyện Mô Hình"
   - Cấu hình tham số huấn luyện
   - Nhấp "🚀 Huấn Luyện Các Mô Hình Đã Chọn"

5. **Đánh Giá Mô Hình**
   - Xem so sánh hiệu suất trong "Đánh Giá & So Sánh Mô Hình"
   - Phân tích chỉ số chi tiết và trực quan hóa

6. **Dự Đoán Thời Gian Thực**
   - Sử dụng "Dự Đoán Thời Gian Thực" để dự báo tức thì
   - Nhập tham số môi trường
   - Nhận dự đoán AQI và tư vấn sức khỏe

### Hướng Dẫn Nhập Tham Số

#### **Nồng Độ Chất Ô Nhiễm**:
- PM2.5: 5-200 μg/m³ (phạm vi điển hình)
- PM10: 10-300 μg/m³
- NO₂: 5-150 μg/m³
- SO₂: 2-100 μg/m³
- CO: 0.5-10 mg/m³
- O₃: 10-200 μg/m³

#### **Tham Số Thời Tiết**:
- Nhiệt độ: -10°C đến 50°C
- Độ ẩm: 30% đến 95%
- Tốc độ gió: 0.5 đến 10 m/s
- Áp suất: 900 đến 1100 hPa
- Lượng mưa: 0 đến 100 mm

## 📊 Kết Quả Mong Đợi

### Hiệu Suất Thuật Toán (Kết Quả Điển Hình)

| Thuật Toán | Nhiệm Vụ | RMSE | R² | Accuracy | F1-Score |
|-----------|------|------|----|----------|----------|
| Hồi Quy Tuyến Tính | Dự Đoán AQI | 18-25 | 0.82-0.88 | - | - |
| Cây Quyết Định | Dự Đoán AQI | 15-22 | 0.85-0.90 | - | - |
| SVM | Phân Loại | - | - | 0.84-0.89 | 0.82-0.87 |
| Hồi Quy Logistic | Phân Loại | - | - | 0.80-0.86 | 0.78-0.84 |

### Trường Hợp Sử Dụng Tốt Nhất

- **Độ Chính Xác Cao Nhất**: SVM cho phân loại
- **Huấn Luyện Nhanh Nhất**: Hồi Quy Tuyến Tính cho hồi quy
- **Dễ Diễn Giải Nhất**: Cây Quyết Định cho phân tích
- **Đáng Tin Cậy Nhất**: Hồi Quy Logistic cho sản xuất

## 🔬 Triển Khai Kỹ Thuật

### Pipeline Tiền Xử Lý Dữ Liệu

1. **Xử Lý Giá Trị Thiếu**
   - Biến số số học: Imputation trung vị
   - Biến số phân loại: Imputation mode

2. **Loại Bỏ Ngoại Lệ**
   - Phương pháp IQR với ngưỡng 1.5×IQR
   - Áp dụng cho tất cả đặc trưng số học ngoại trừ mục tiêu

3. **Kỹ Thuật Đặc Trưng**
   - Đặc trưng thời gian: Giờ, ngày trong tuần, tháng, mùa
   - Mã hóa tuần hoàn: biến đổi sin/cos
   - Thuật ngữ tương tác: tỷ lệ chất ô nhiễm, tương tác thời tiết
   - Chỉ số tổng hợp: chỉ số ô nhiễm giao thông/công nghiệp

4. **Chuẩn Hóa Dữ Liệu**
   - StandardScaler cho đặc trưng số học
   - Mã hóa nhãn cho biến phân loại

### Quy Trình Huấn Luyện Mô Hình

1. **Chia Dữ Liệu**
   - 80% huấn luyện, 20% kiểm tra
   - Lấy mẫu phân tầng cho phân loại

2. **Cross-Validation**
   - 5-fold CV để đánh giá robust
   - Ngăn chặn overfitting

3. **Tinh Chỉnh Siêu Tham Số**
   - Grid Search CV (tùy chọn)
   - Tối ưu hóa cho RMSE (hồi quy) hoặc accuracy (phân loại)

## 🚀 Cải Tiến Tương Lai

### **Cải Tiến Kỹ Thuật**
- **Học Sâu**: LSTM/GRU cho dự đoán chuỗi thời gian
- **Phương Pháp Ensemble**: Random Forest, Gradient Boosting
- **Lựa Chọn Đặc Trưng**: Loại bỏ đặc trưng đệ quy
- **Tối Ưu Hóa Siêu Tham Số**: Tối ưu hóa Bayesian

### **Cải Tiến Dữ Liệu**
- **Tích Hợp Dữ Liệu Thực**: Kết nối trạm giám sát thực tế
- **Mở Rộng Địa Lý**: Bao gồm các thành phố khác của Việt Nam
- **Đặc Trưng Thêm**: Dữ liệu giao thông, phát thải công nghiệp
- **Độ Phân Giải Thời Gian**: Streaming dữ liệu thời gian thực

### **Cải Tiến Ứng Dụng**
- **Ứng Dụng Di Động**: Ứng dụng iOS/Android
- **Phát Triển API**: RESTful API để tích hợp
- **Hệ Thống Cảnh Báo**: Cảnh báo ô nhiễm tự động
- **Phân Tích Lịch Sử**: Phân tích xu hướng dài hạn

## 🏆 Thành Tựu Dự Án

### **Đóng Góp Học Thuật**
- ✅ So sánh toàn diện thuật toán học máy
- ✅ Tạo bộ dữ liệu giả lập thực tế
- ✅ Pipeline tiền xử lý hoàn chỉnh
- ✅ Framework đánh giá robust

### **Ứng Dụng Thực Tiễn**
- ✅ Giao diện web tương tác
- ✅ Khả năng dự đoán thời gian thực
- ✅ Hệ thống tư vấn sức khỏe
- ✅ Tối ưu hóa hiệu suất

### **Xuất Sắc Kỹ Thuật**
- ✅ Code module, dễ bảo trì
- ✅ Tài liệu toàn diện
- ✅ Xử lý lỗi và xác thực
- ✅ Kiến trúc có khả năng mở rộng

## 👥 Đóng Góp Thành Viên

| Thành Viên | Thuật Toán | Trách Nhiệm |
|--------|-----------|------------------|
| Mạnh | Hồi Quy Tuyến Tính | Triển khai thuật toán, nền tảng toán học |
| Quang | Cây Quyết Định (CART) | Phương pháp cây, phân tích tầm quan trọng đặc trưng |
| Tiến | SVM | Support vector machines, tối ưu hóa kernel |
| Thương | Hồi Quy Logistic | Thuật toán phân loại, mô hình xác suất |

## 📞 Hỗ Trợ & Liên Hệ

Để có câu hỏi, vấn đề, hoặc đóng góp:

1. **Tài liệu**: Tham khảo README và chú thích code
2. **Vấn đề**: Kiểm tra output console để biết thông báo lỗi
3. **Gỡ Lỗi**: Sử dụng công cụ khám phá dữ liệu tích hợp
4. **Hiệu Suất**: Theo dõi thời gian huấn luyện và sử dụng bộ nhớ

## 📄 Giấy Phép

Dự án này dành cho mục đích giáo dục và nghiên cứu. Vui lòng trích dẫn phù hợp nếu sử dụng trong công việc học thuật.

## 🙏 Lời Cảm Ơn

- **Scikit-learn**: Thuật toán học máy
- **Streamlit**: Framework ứng dụng web
- **Plotly**: Trực quan hóa tương tác
- **Pandas**: Thao tác dữ liệu
- **NumPy**: Tính toán số học

---

**🎉 Cảm ơn bạn đã sử dụng Hệ Thống Dự Đoán Ô Nhiễm Không Khí Hà Nội!**

Dự án này thể hiện ứng dụng thực tế của học máy trong giám sát môi trường và bảo vệ sức khỏe công chúng. So sánh toàn diện các thuật toán cung cấp thông tin giá trị cho cả nghiên cứu học thuật và triển khai thực tế.
