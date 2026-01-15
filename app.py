import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import os
import urllib.request
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# ================================
# Download model files from HuggingFace
# ================================
HF_BASE = "https://huggingface.co/satya-anand-ml/churn-ann-model/resolve/main/"

def download(file):
    if not os.path.exists(file):
        url = HF_BASE + file + "?download=true"
        st.write(f"Downloading {file}...")
        urllib.request.urlretrieve(url, file)

download("ann_churn_model.h5")
download("scaler.pkl")
download("label_encoder_gender.pkl")
download("onehot_encoder_geo.pkl")



# ================================
# Load Model & Encoders
# ================================

model = tf.keras.models.load_model("ann_churn_model.h5")

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ================================
# UI Styling
# ================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1 {
    text-align: center;
    color: #00ffd5;
    font-size: 42px;
}
label {
    color: #ffffff !important;
    font-weight: 600;
}
input, select, textarea {
    background-color: #1e2a38 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #00ffd5 !important;
}
button {
    background: linear-gradient(90deg, #00ffd5, #00c3ff) !important;
    color: black !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}
div.stAlert {
    background-color: #1e2a38 !important;
    border-radius: 12px !important;
    border: 1px solid #00ffd5;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ================================
# App UI
# ================================

st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_number = st.selectbox("Is Active Member", [0, 1])

# ================================
# Prepare Input
# ================================

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_number],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

# ================================
# Prediction
# ================================

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"### Churn Probability: **{prediction_proba:.2f}**")

if prediction_proba > 0.5:
    st.error("⚠️ The customer is likely to churn.")
else:
    st.success("✅ The customer is not likely to churn.")
