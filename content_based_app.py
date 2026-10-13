import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper: ensure samples.csv exists; if not, create a small sample dataset
SAMPLES_PATH = 'samples.csv'
PICKLE_PATH = 'companies_cosine_sim.pkl'
BANNER_PATH = 'channels4_banner.jpg'

def create_sample_csv(path=SAMPLES_PATH):
    sample_data = [
        {
            'id': 1,
            'Company Name': 'VietTech Solutions',
            'Company Type': 'IT Outsourcing',
            'Company industry': 'Software Development',
            'Company size': '50-200',
            'Country': 'Vietnam',
            'Working days': 'Mon-Fri',
            'Overtime Policy': 'Sometimes',
            'Company overview': 'VietTech Solutions is an outsourcing company focusing on web and mobile applications using Python and JavaScript. We deliver scalable solutions for international clients.',
            'Our key skills': 'Python, Django, React Native',
            "Why you'll love working here": 'Flexible hours, friendly team',
            'Location': 'Hanoi',
            'Href': 'https://example.com/viettech'
        },
        {
            'id': 2,
            'Company Name': 'Saigon AI Lab',
            'Company Type': 'IT Product',
            'Company industry': 'Artificial Intelligence',
            'Company size': '20-50',
            'Country': 'Vietnam',
            'Working days': 'Mon-Fri',
            'Overtime Policy': 'Rarely',
            'Company overview': 'Saigon AI Lab builds machine learning products for computer vision and NLP. We prototype rapidly and value data-driven decisions.',
            'Our key skills': 'Machine Learning, Python, TensorFlow',
            "Why you'll love working here": 'Cutting-edge projects, mentorship',
            'Location': 'Ho Chi Minh City',
            'Href': 'https://example.com/saigonai'
        },
        {
            'id': 3,
            'Company Name': 'Hue Cloud Services',
            'Company Type': 'Cloud Provider',
            'Company industry': 'Cloud & DevOps',
            'Company size': '10-30',
            'Country': 'Vietnam',
            'Working days': 'Mon-Sat',
            'Overtime Policy': 'Sometimes',
            'Company overview': 'Hue Cloud Services provides cloud architecture, DevOps automation and consultancy. We help teams ship faster using CI/CD and Kubernetes.',
            'Our key skills': 'AWS, Kubernetes, Terraform',
            "Why you'll love working here": 'Ownership, learning',
            'Location': 'Hue',
            'Href': 'https://example.com/huecloud'
        },
        {
            'id': 4,
            'Company Name': 'Danang Web Studio',
            'Company Type': 'Agency',
            'Company industry': 'Web Development',
            'Company size': '5-20',
            'Country': 'Vietnam',
            'Working days': 'Mon-Fri',
            'Overtime Policy': 'Often',
            'Company overview': 'Danang Web Studio is a small agency building websites and e-commerce platforms, specializing in frontend experiences and responsive design.',
            'Our key skills': 'HTML, CSS, JavaScript',
            "Why you'll love working here": 'Creative projects, close-knit team',
            'Location': 'Da Nang',
            'Href': 'https://example.com/danang'
        },
        {
            'id': 5,
            'Company Name': 'Hanoi DataOps',
            'Company Type': 'Consulting',
            'Company industry': 'Data Engineering',
            'Company size': '30-80',
            'Country': 'Vietnam',
            'Working days': 'Mon-Fri',
            'Overtime Policy': 'Sometimes',
            'Company overview': 'Hanoi DataOps focuses on data pipelines, analytics and business intelligence. We help companies make sense of large datasets.',
            'Our key skills': 'SQL, Python, Airflow',
            "Why you'll love working here": 'Impact, strong data culture',
            'Location': 'Hanoi',
            'Href': 'https://example.com/hanoidata'
        }
    ]
    df = pd.DataFrame(sample_data)
    df.to_csv(path, index=False)
    return df

# Load or create samples.csv
if not os.path.exists(SAMPLES_PATH):
    st.warning(f"{SAMPLES_PATH} not found — creating a sample file with {5} companies.")
    df_companies = create_sample_csv(SAMPLES_PATH)
else:
    df_companies = pd.read_csv(SAMPLES_PATH)

# Ensure 'Company overview' exists and is string
if 'Company overview' not in df_companies.columns:
    df_companies['Company overview'] = ''
else:
    df_companies['Company overview'] = df_companies['Company overview'].fillna('')

# Build or load cosine similarity pickle
if os.path.exists(PICKLE_PATH):
    try:
        with open(PICKLE_PATH, 'rb') as f:
            cosine_sim_new = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load {PICKLE_PATH}: {e}. Will recompute.")
        cosine_sim_new = None
else:
    cosine_sim_new = None

if cosine_sim_new is None:
    # Compute TF-IDF + cosine similarity
    st.info('Computing TF-IDF matrix and cosine similarity (this runs once).')
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_companies['Company overview'])
    cosine_sim_new = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Save for future runs
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(cosine_sim_new, f)

# Function cần thiết
def get_recommendations(df, id, cosine_sim, nums=5):
    # Get the index of the company that matches the id
    matching_indices = df.index[df['id'] == id].tolist()
    if not matching_indices:
        # Return an empty DataFrame if no match
        return pd.DataFrame()
    idx = matching_indices[0]
    # Get the pairwise similarity scores of all companies with that company
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the companies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the nums most similar companies (Ignoring the company itself)
    sim_scores = sim_scores[1:nums+1]
    # Get the company indices
    company_indices = [i[0] for i in sim_scores]

    # Return the top n most similar companies as a DataFrame
    return df.iloc[company_indices]

# Hiển thị đề xuất ra bảng
def display_recommended_companies(recommended_companies, cols=5):
    for i in range(0, len(recommended_companies), cols):
        cols_display = st.columns(cols)
        for j, col in enumerate(cols_display):
            if i + j < len(recommended_companies):
                company = recommended_companies.iloc[i + j]
                with col:
                    st.write(company.get('Company Name', 'N/A'))
                    expander = st.expander("Company overview")
                    company_description = company.get('Company overview', '')
                    truncated_description = ' '.join(str(company_description).split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")

###### Giao diện Streamlit ######
if os.path.exists(BANNER_PATH):
    try:
        st.image(BANNER_PATH, use_container_width=True)
    except Exception:
        st.title('itViet_Project - Company Recommendation System')
else:
    st.title('itViet_Project - Company Recommendation System')

# Kiểm tra xem 'random_companies' đã có trong session_state hay chưa
if 'random_companies' not in st.session_state:
    st.session_state.random_companies = df_companies.sample(n=min(10, len(df_companies)), random_state=42)

# Kiểm tra xem 'selected_id' đã có trong session_state hay chưa
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None

# Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
company_options = [(row['Company Name'], row['id']) for index, row in st.session_state.random_companies.iterrows()]
# Tạo một dropdown với options là các tuple này
selected_company = st.selectbox(
    "Chọn công ty",
    options=company_options,
    format_func=lambda x: x[0]  # Hiển thị tên công ty
)
# Display the selected company
st.write("Bạn đã chọn:", selected_company[0])

# Cập nhật session_state dựa trên lựa chọn hiện tại
st.session_state.selected_id = selected_company[1]

if st.session_state.selected_id is not None:
    st.write("id: ", st.session_state.selected_id)
    # Hiển thị thông tin sản phẩm được chọn
    selected_company_info = df_companies[df_companies['id'] == st.session_state.selected_id]

    if not selected_company_info.empty:
        st.write('### ', selected_company_info['Company Name'].values[0])

        company_description = selected_company_info['Company overview'].values[0]
        truncated_description = ' '.join(str(company_description).split()[:100])
        st.write('##### Information:')
        st.write(truncated_description, '...')

        st.write('##### Các công ty liên quan:')
        recommendations = get_recommendations(df_companies, st.session_state.selected_id, cosine_sim=cosine_sim_new, nums=3)
        if recommendations.empty:
            st.write('Không có đề xuất.')
        else:
            display_recommended_companies(recommendations, cols=3)
    else:
        st.write(f"Không tìm thấy công ty với ID: {st.session_state.selected_id}")
