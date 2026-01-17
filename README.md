
# 📌 Customer Churn Prediction using Artificial Neural Network (ANN)

## 📖 Project Overview
Customer churn is a major challenge for banks and financial institutions.  
This project predicts whether a customer is **likely to churn (leave the bank)** or **stay loyal** using an **Artificial Neural Network (ANN)**.

The project focuses on:
1. ANN-based churn prediction  
2. Real-time inference using a trained model  
3. Model deployment using **Streamlit**  

The trained model and preprocessing files are hosted on **Hugging Face Hub**, making the application lightweight and production-ready.

---

## 🧠 Problem Statement
Build a machine learning system that predicts customer churn based on demographic and financial features such as:

- Credit Score  
- Age  
- Balance  
- Tenure  
- Number of Products  
- Credit Card Status  
- Active Membership  
- Estimated Salary  
- Geography  
- Gender  

---

## 🏗️ Project Structure
```

Customer-Churn-Deploy/
│
├── app.py
├── requirements.txt
└── README.md

```

Model files are downloaded dynamically from **Hugging Face Hub** at runtime.

---

## ⚙️ ANN Model Details
- **Model Type:** Artificial Neural Network (ANN)  
- **Framework:** TensorFlow / Keras  
- **Task:** Binary Classification  
- **Output:** Churn Probability (0–1)  

Saved model artifacts:
- ann_churn_model.h5  
- scaler.pkl  
- label_encoder_gender.pkl  
- onehot_encoder_geo.pkl  

---

## 🔮 Customer Churn Prediction Logic
1. User inputs customer details via Streamlit UI  
2. Categorical features are encoded  
3. Numerical features are scaled  
4. Processed data is passed to the ANN model  
5. Churn probability and result are displayed  

Decision rule:
- Probability > 0.5 → Customer likely to churn  
- Probability ≤ 0.5 → Customer not likely to churn  

---

## 🚀 Model Deployment using Streamlit
The application is built using **Streamlit** and provides:

- Interactive sliders and dropdowns  
- Clean and modern UI  
- Real-time churn probability prediction  
- Clear success and warning messages  

Live App:  
https://customer-churn-deploy-7ztmntzrzfnsscltzscftk.streamlit.app/

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries & Frameworks:**
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - TensorFlow / Keras  
  - Streamlit  
  - Hugging Face Hub  
- **Model:** Artificial Neural Network (ANN)  
- **Version Control:** Git & GitHub  

---

## 📦 Hugging Face Model Repository
```

satya-anand-ml/churn-ann-model

````

Contains:
- Trained ANN model  
- Feature scaler  
- Label encoder  
- One-hot encoder  

---

## ▶️ How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/satya-anand-ml/Customer-Churn-Deploy.git
cd Customer-Churn-Deploy
````

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📌 Future Enhancements

* Add model explainability (SHAP / LIME)
* Improve UI/UX
* Cloud deployment optimization
* Compare ANN with other ML models

---

## 👤 Author

**Satya Anand**

📧 Email: [satyaanand442@gmail.com](mailto:satyaanand442@gmail.com)
🔗 LinkedIn: [https://www.linkedin.com/in/satya-anand-25122003k](https://www.linkedin.com/in/satya-anand-25122003k)
🐙 GitHub: [https://github.com/satya2337](https://github.com/satya2337)

---

## ⭐ Acknowledgement

Thanks to open-source datasets, TensorFlow, Streamlit, and Hugging Face
for making this project possible.



