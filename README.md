# 📌 Customer Churn Prediction using Artificial Neural Network (ANN)

## 📖 Project Overview
Customer churn is one of the major challenges faced by banks and financial institutions.  
This project predicts whether a customer is **likely to leave the bank (churn)** or **stay loyal** using an **Artificial Neural Network (ANN)**.

The project is divided into **three major parts**:

1. ANN Model Training  
2. Customer Churn Prediction  
3. Model Deployment using Streamlit  

This solution helps banks take **proactive decisions** to retain customers by identifying high-risk churn cases.

---

## 🧠 Problem Statement
Build a machine learning model that predicts customer churn based on demographic and financial details such as:

- Credit Score  
- Age  
- Balance  
- Tenure  
- Number of Products  
- Active Membership  
- Estimated Salary  

---

## 🏗️ Project Structure

```
Customer-Churn-ANN/
│
├── data/
│   └── churn_dataset.csv
│
├── model/
│   ├── ann_churn_model.h5
│   ├── scaler.pkl
│   ├── label_encoder_gender.pkl
│   └── onehot_encoder_geo.pkl
│
├── src/
│   ├── train_ann_model.py
│   └── prediction.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Part 1: ANN Model Training
Model training and preprocessing are handled in **`train_ann_model.py`**, which includes:

- Data loading from `churn_dataset.csv`  
- Label Encoding for Gender  
- One-Hot Encoding for Geography  
- Feature scaling using `StandardScaler`  
- Splitting data into training and testing sets  
- Building and training the ANN  
- Saving:
  - Trained model → `ann_churn_model.h5`
  - Scaler → `scaler.pkl`
  - Encoders → `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`

---

## 🔮 Part 2: Customer Churn Prediction
Prediction logic is implemented in **`prediction.py`**, which:

- Loads the trained ANN model  
- Loads the saved encoders and scaler  
- Applies the same preprocessing to new customer input  
- Predicts whether the customer will:
  - **Stay Loyal**
  - **Exit (Churn)**

---

## 🚀 Part 3: Model Deployment using Streamlit
The web application is built using **Streamlit** in **`app.py`**, which:

- Provides a user-friendly UI  
- Takes customer details as input (sliders & dropdowns)  
- Sends the input to `prediction.py`  
- Displays churn prediction in real-time  

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - TensorFlow / Keras  
- **Model Type:** Artificial Neural Network (ANN)  
- **Web Framework:** Streamlit  
- **IDE & Tools:** VS Code, Jupyter Notebook  
- **Version Control:** Git & GitHub  

---

## 📊 Dataset
- **Bank Customer Churn Dataset** (`churn_dataset.csv`)  
- Contains customer demographic and financial information  
- Target variable:
  - `1` → Customer Exited  
  - `0` → Customer Stayed  

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/satya2337/Customer-Churn-Prediction-App.git
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Train the Model
If you want to retrain the ANN model:
```bash
python train_ann_model.py
```

This will generate:
- `ann_churn_model.h5`
- `scaler.pkl`
- `label_encoder_gender.pkl`
- `onehot_encoder_geo.pkl`

### Step 4: Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📌 Future Enhancements
- Hyperparameter tuning for better accuracy  
- Display model accuracy and metrics in the UI  
- Deploy the app on cloud platforms (Heroku / AWS / Streamlit Cloud)  
- Compare ANN with other ML models  

---

## 👤 Author
**Satya Anand**  
📧 Email: [satyaanand442@gmail.com](mailto:satyaanand442@gmail.com)  
🔗 LinkedIn: https://www.linkedin.com/in/satya-anand-25122003k  
🐙 GitHub: https://github.com/satya2337  

---

## ⭐ Acknowledgement
Thanks to open-source datasets and libraries that made this project possible.
