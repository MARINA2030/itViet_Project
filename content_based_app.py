import streamlit as st
import pandas as pd
import pickle

# function cần thiết
def get_recommendations(df, id, cosine_sim, nums=5):
    # Get the index of the company that matches the id
    matching_indices = df.index[df['id'] == id].tolist()
    if not matching_indices:
        print(f"No company found with ID: {id}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
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
                    st.write(company['Company Name'])                  
                    expander = st.expander(f"Company overview")
                    company_description = company['Company overview']
                    truncated_description = ' '.join(company_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

# Đọc dữ liệu sản phẩm
df_companies = pd.read_csv('samples.csv')
if 'random_companies' not in st.session_state:
    df_companies = pd.read_csv('samples.csv')
    st.session_state.random_companies = df_companies.sample(n=10, random_state=42)

# Open and read file to cosine_sim_new
with open('companies_cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

###### Giao diện Streamlit ######
st.image('channels4_banner.jpg', use_container_width=True) # phiên bản mới hơn

# Kiểm tra xem 'selected_id' đã có trong session_state hay chưa
if 'selected_id' not in st.session_state:
    # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
    st.session_state.selected_id = None

# Theo cách cho người dùng chọn công ty từ dropdown
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

if st.session_state.selected_id:
    st.write("id: ", st.session_state.selected_id)
    # Hiển thị thông tin sản phẩm được chọn
    selected_company_info = df_companies[df_companies['id'] == st.session_state.selected_id]

    if not selected_company_info.empty:
        st.write('### ', selected_company_info['Company Name'].values[0])

        company_description = selected_company_info['Company overview'].values[0]
        truncated_description = ' '.join(company_description.split()[:100])
        st.write('##### Information:')
        st.write(truncated_description, '...')

        st.write('##### Các công ty liên quan:')
        recommendations = get_recommendations(df_companies, st.session_state.selected_id, cosine_sim=cosine_sim_new, nums=3) 
        display_recommended_companies(recommendations, cols=3)
    else:
        st.write(f"Không tìm thấy công ty với ID: {st.session_state.selected_id}")