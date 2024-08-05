import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Load external CSS for styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load medicine dataframe and similarity vector from pickles
medicines_dict = pickle.load(open('medicine_dict.pkl', 'rb'))
medicines = pd.DataFrame(medicines_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Function to recommend alternative medicines
def recommend(medicine):
    if medicine not in medicines['Drug_Name'].values:
        st.error("Selected medicine not found. Please choose another.")
        return []
    
    medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_medicines = [medicines.iloc[i[0]].Drug_Name for i in medicines_list]
    return recommended_medicines

# Application Frontend
st.title('Medicine Recommender System')

# Searchbox to select a medicine
selected_medicine_name = st.selectbox(
    'Type the name of the medicine to get Alternate recommendations',
    medicines['Drug_Name'].values
)

# Recommendation Program
if st.button('Recommend Medicine'):
    recommendations = recommend(selected_medicine_name)
    if recommendations:
        st.write("Recommended Alternative Medicines:")
        for i, recommended_med in enumerate(recommendations, start=1):
            st.write(f"{i}. {recommended_med}")
            # Add purchase links for recommended medicines
            st.write(f"Purchase: [PharmEasy](https://pharmeasy.in/search/all?name={recommended_med})")
    else:
        st.warning("No recommendations found.")




