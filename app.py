import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
from huggingface_hub import hf_hub_download

# ------------------- UI THEME -------------------
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

# ------------------- HUGGINGFACE MODEL LOADING -------------------
MODEL_REPO = "satya-anand-ml/churn-ann-model"

model_path = hf_hub_download(MODEL_REPO, "ann_churn_model.h5")
geo_path = hf_hub_download(MODEL_REPO, "onehot_encoder_geo.pkl")
gender_path = hf_hub_download(MODEL_REPO, "label_encoder_gender.pkl")
scaler_path = hf_hub_download(MODEL_REPO, "scaler.pkl")

model = tf.keras.models.load_model(model_path)

with open(geo_path, "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open(gender_path, "rb") as f:
    label_encoder_gender = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# ------------------- APP UI -------------------
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])

# ------------------- INPUT PROCESSING -------------------
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_data)

# ------------------- PREDICTION -------------------
prediction = model.predict(input_scaled)
prob = prediction[0][0]

st.subheader(f"Churn Probability: {prob:.2f}")

if prob > 0.5:
    st.error("⚠️ Customer is likely to churn")
else:
    st.success("✅ Customer is not likely to churn")
