import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
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


if not os.path.exists(SAMPLES_PATH):
    st.warning(f"{SAMPLES_PATH} not found — creating a sample file with 5 companies.")
    df_companies = create_sample_csv(SAMPLES_PATH)
else:
    try:
        df_companies = pd.read_csv(SAMPLES_PATH)
    except Exception:
        logger.exception("Failed to read samples.csv with default encoding, recreating sample CSV")
        st.warning("Failed to read samples.csv — recreating sample data.")
        df_companies = create_sample_csv(SAMPLES_PATH)

if 'Company overview' not in df_companies.columns:
    df_companies['Company overview'] = ''
else:
    df_companies['Company overview'] = df_companies['Company overview'].fillna('')

if df_companies['Company overview'].astype(str).str.strip().eq('').all():
    logger.warning('All Company overview fields are empty — inserting a placeholder.')
    df_companies['Company overview'] = 'No description available'

cosine_sim_new = None
if os.path.exists(PICKLE_PATH):
    try:
        with open(PICKLE_PATH, 'rb') as f:
            cosine_sim_new = pickle.load(f)
    except Exception as e:
        logger.exception(f"Failed to load {PICKLE_PATH}: {e}. Will recompute.")
        st.error(f"Failed to load {PICKLE_PATH}: {e}. Will recompute.")
        cosine_sim_new = None

if cosine_sim_new is None:
    st.info('Computing TF-IDF matrix and cosine similarity (this runs once).')
    try:
        vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_companies['Company overview'].astype(str))
        cosine_sim_new = cosine_similarity(tfidf_matrix, tfidf_matrix)
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(cosine_sim_new, f)
    except ValueError as e:
        logger.exception('TF-IDF failed: %s', e)
        st.error('Could not compute TF-IDF; using fallback similarity matrix.')
        cosine_sim_new = np.eye(len(df_companies))


def get_recommendations(df, company_id, cosine_sim, nums=5):
    matching_indices = df.index[df['id'] == company_id].tolist()
    if not matching_indices:
        return pd.DataFrame()
    idx = matching_indices[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums+1]
    company_indices = [i[0] for i in sim_scores]
    return df.iloc[company_indices]


def display_recommended_companies(recommended_companies, cols=3):
    cols = max(1, cols)
    for i in range(0, len(recommended_companies), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < len(recommended_companies):
                company = recommended_companies.iloc[i + j]
                with col:
                    st.subheader(company.get('Company Name', 'N/A'))
                    expander = st.expander('Company overview')
                    description = str(company.get('Company overview', ''))
                    truncated = ' '.join(description.split()[:100]) + ('...' if len(description.split()) > 100 else '')
                    expander.write(truncated)


if os.path.exists(BANNER_PATH):
    try:
        st.image(BANNER_PATH, use_container_width=True)
    except Exception:
        st.title('itViet_Project - Company Recommendation System')
else:
    st.title('itViet_Project - Company Recommendation System')

if 'random_companies' not in st.session_state:
    st.session_state.random_companies = df_companies.sample(n=min(10, len(df_companies)), random_state=42)

if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None

company_options = [(row['Company Name'], int(row['id'])) for _, row in st.session_state.random_companies.iterrows()]
selected_company = st.selectbox(
    'Chọn công ty',
    options=company_options,
    format_func=lambda x: x[0],
    key='company_select'
)

st.write('Bạn đã chọn:', selected_company[0])
st.session_state.selected_id = selected_company[1]

if st.session_state.selected_id is not None:
    st.write('id:', st.session_state.selected_id)
    selected_company_info = df_companies[df_companies['id'] == st.session_state.selected_id]

    if not selected_company_info.empty:
        st.write('###', selected_company_info['Company Name'].values[0])
        description = str(selected_company_info['Company overview'].values[0])
        truncated = ' '.join(description.split()[:100]) + ('...' if len(description.split()) > 100 else '')
        st.write('##### Information:')
        st.write(truncated)

        st.write('##### Các công ty liên quan:')
        recommendations = get_recommendations(df_companies, st.session_state.selected_id, cosine_sim=cosine_sim_new, nums=3)
        if recommendations.empty:
            st.write('Không có đề xuất.')
        else:
            display_recommended_companies(recommendations, cols=3)
    else:
        st.write(f"Không tìm thấy công ty với ID: {st.session_state.selected_id}")
