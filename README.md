# itViet_Project - Company Recommendation System

## Giới thiệu
Đây là một ứng dụng **Recommendation System** dành cho các công ty IT, giúp tìm kiếm và đề xuất các công ty liên quan dựa trên nội dung mô tả công ty.

## Chức năng chính

### 1. Tìm kiếm công ty
- Hiển thị danh sách công ty ngẫu nhiên
- Chọn công ty từ dropdown
- Xem thông tin chi tiết của công ty

### 2. Đề xuất công ty liên quan
- Sử dụng **Cosine Similarity** để tính độ tương đồng giữa các công ty
- Dựa trên **Company Overview** của mỗi công ty
- Hiển thị top 3 công ty tương tự nhất

### 3. Xem chi tiết công ty
- Tên công ty
- Mô tả tóm tắt (100 từ đầu tiên)
- Các công ty liên quan

## Cài đặt và chạy

### Yêu cầu
- Python 3.7+
- pip

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Chạy ứng dụng
```bash
streamlit run content_based_app.py
```

Ứng dụng sẽ chạy tại `http://localhost:8501`

## Cấu trúc dự án

```
itViet_Project/
├── content_based_app.py          # Ứng dụng Streamlit chính
├── requirements.txt              # Dependencies
├── setup.sh                      # Script cấu hình Streamlit
├── samples.csv                   # Dữ liệu công ty (cần thêm)
├── companies_cosine_sim.pkl      # Ma trận độ tương đồng (cần tính toán)
├── channels4_banner.jpg          # Banner hình ảnh (cần thêm)
└── README.md                     # File này
```

## Dữ liệu

### File `samples.csv`
Phải chứa các cột:
- `id`: ID công ty
- `Company Name`: Tên công ty
- `Company Type`: Loại công ty (IT Outsourcing, IT Product, etc.)
- `Company industry`: Ngành công ty
- `Company size`: Quy mô công ty
- `Country`: Quốc gia
- `Working days`: Ngày làm việc
- `Overtime Policy`: Chính sách làm thêm
- `Company overview`: Mô tả chi tiết công ty
- `Our key skills`: Kỹ năng chính
- `Why you'll love working here`: Lợi ích làm việc
- `Location`: Địa chỉ
- `Href`: Link công ty

### File `companies_cosine_sim.pkl`
Ma trận độ tương đồng được tính từ các mô tả công ty.

Tính toán:
```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vector hóa nội dung
vectorizer = TfidfVectorizer(analyzer='word')
tfidf_matrix = vectorizer.fit_transform(df['Company overview'])

# Tính toán độ tương đồng
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Lưu vào file
with open('companies_cosine_sim.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)
```

## API/Hàm chính

### `get_recommendations(df, id, cosine_sim, nums=5)`
Tìm các công ty tương tự nhất.

**Parameters:**
- `df`: DataFrame chứa dữ liệu công ty
- `id`: ID công ty cần tìm đề xuất
- `cosine_sim`: Ma trận độ tương đồng
- `nums`: Số lượng công ty muốn đề xuất (mặc định: 5)

**Returns:**
- DataFrame chứa các công ty được đề xuất

### `display_recommended_companies(recommended_companies, cols=5)`
Hiển thị danh sách công ty theo bố cục grid.

**Parameters:**
- `recommended_companies`: DataFrame công ty được đề xuất
- `cols`: Số cột hiển thị (mặc định: 5)

## Công nghệ sử dụng

- **Streamlit**: Framework tạo web app
- **Pandas**: Xử lý dữ liệu
- **Scikit-learn**: Machine Learning (TF-IDF, Cosine Similarity)
- **Matplotlib/Seaborn**: Visualization

## Tác giả
- MARINA2030

## License
MIT