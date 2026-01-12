# DoAn-Nhom8

## Giới thiệu
Dự án học máy sử dụng các thuật toán hồi quy (Regression) nhằm dự đoán:
- Mức độ nghiêm trọng của thảm họa (severity_index)
- Thời gian phản hồi cứu trợ (response_time_hours)

Dữ liệu sử dụng: Global Disaster Response 2018–2024.

## Môi trường yêu cầu
- Python >= 3.8
- Pip (Python Package Manager)

## Cài đặt
1. Clone repository:
   git clone <repository_url>
   cd DoAn-Nhom8

2. Tạo môi trường ảo:
   - Windows:
     python -m venv venv
     venv\Scripts\activate
   - MacOS/Linux:
     python3 -m venv venv
     source venv/bin/activate

3. Cài đặt thư viện:
   - Nếu có requirements.txt:
     pip install -r requirements.txt
   - Nếu không:
     pip install numpy pandas scikit-learn matplotlib seaborn

## Mô hình sử dụng
- Random Forest Regression: dự đoán severity_index (mô hình chính, có tuning)
- Linear Regression: dự đoán response_time_hours (baseline)

## Cách chạy chương trình
Chạy pipeline chính:
Chương trình sẽ thực hiện:
- Tiền xử lý dữ liệu
- Phân tích EDA
- Huấn luyện và tuning mô hình
- Đánh giá mô hình
- Nhập dữ liệu người dùng và trả ra kết quả dự đoán

## Kết quả
- Random Forest cho độ chính xác cao hơn Linear Regression
- Các đặc trưng ảnh hưởng mạnh: economic_loss_usd, casualties, aid_amount_usd

## Tác giả
Nhóm 8 – Môn Học Máy
