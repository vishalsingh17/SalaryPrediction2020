import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

dtr_loaded = data["model"]
label_country = data["label_country"]
label_edu = data["label_edu"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some of your information to predict""")

    countries = ("United States", "India", "United Kingdom", "Germany", "Canada", "Brazil", "France", "Spain", "Australia", "Netherlands", "Poland", "Italy", "Russian Federation", "Sweden")
    education = ('Less than a Bachelors', "Bachelor's degree", 'Master’s degree', 'Post grad')

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X1 = np.array([[country, education, experience]])
        X1[:, 0] = label_country.transform(X1[:, 0])
        X1[:, 1] = label_edu.transform(X1[:, 1])
        X1 = X1.astype(float)

        salary = dtr_loaded.predict(X1)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")