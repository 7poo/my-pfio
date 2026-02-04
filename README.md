# 🎓 Student Credit & Success Agent

Dự án Dashboard theo dõi và AI Agent tư vấn tín chỉ cho sinh viên, giúp tối ưu hóa kết quả học tập và cảnh báo rủi ro sớm.

## 🚀 Tính Năng Chính
- **Dashboard Monitoring**: Theo dõi KPI, xu hướng điểm số, tỷ lệ hoàn thành theo học kỳ và ngành.
- **AI Recommendation**: Hệ thống gợi ý số tín chỉ đăng ký tối ưu dựa trên năng lực và lịch sử học tập cá nhân (Tính năng nổi bật: "Safety Heuristics" đảm bảo an toàn & khả thi).
- **Optimization Insights**: Phân tích các yếu tố ảnh hưởng đến kết quả học tập.

## 📦 Hướng Dẫn Cài Đặt & Sử Dụng

Bạn có thể chạy dự án này bằng **Docker** (khuyên dùng) hoặc môi trường **Python** thông thường.

### Cách 1: Sử Dụng Docker (Đóng gói sẵn)

**Bước 1: Build Docker Image**
Mở terminal tại thư mục dự án và chạy:
```bash
docker build -t student-agent .
```

**Bước 2: Chạy Container**
```bash
docker run -p 8501:8501 student-agent
```

**Bước 3: Truy cập**
Mở trình duyệt và vào địa chỉ: `http://localhost:8501`

---

### Cách 2: Chạy Trực Tiếp (Python)

**Yêu cầu**: Python 3.8 trở lên.

**Bước 1: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

**Bước 2: Chạy ứng dụng**
```bash
python -m streamlit run app.py
```

---


## 📂 Cấu Trúc Thư Mục
```
data_dashboard_agent/
├── app.py                  # Code chính của ứng dụng Streamlit
├── modules/                # Các module xử lý dữ liệu bổ trợ
├── Dockerfile              # Cấu hình Docker
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── *.pkl                   # File Model & Data đã huấn luyện (Artifacts)
└── README.md               # Hướng dẫn sử dụng này
```

## 📝 Ghi Chú Về Logic AI
Hệ thống sử dụng model ML kết hợp với bộ quy tắc **Safety Heuristics V4**:
- **< 16 tín chỉ**: Tăng độ tin cậy (+1.5%/tín) để khuyến khích giảm tải khi gặp khó khăn.
- **16-22 tín chỉ**: Vùng "Bình thường" (Sweet Spot), tuân theo dự đoán gốc của AI.
- **> 22 tín chỉ**: Phạt dần đều.
- **> 26 tín chỉ**: Phạt cực mạnh để ngăn chặn khuyến nghị quá sức.
