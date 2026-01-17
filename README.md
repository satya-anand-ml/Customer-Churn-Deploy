
---

```markdown
# 📊 Customer Churn Prediction App (ANN + Streamlit)

A machine learning web application that predicts whether a customer is likely to churn using an Artificial Neural Network (ANN). The application is built with Streamlit, TensorFlow, and models hosted on Hugging Face Hub.

## 🚀 Live Demo
https://customer-churn-deploy-7ztmntzrzfnsscltzscftk.streamlit.app/

## 🧠 Model Overview
- Algorithm: Artificial Neural Network (ANN)
- Framework: TensorFlow / Keras
- Problem Type: Binary Classification
- Output: Churn Probability (0–1)

## 🛠️ Tech Stack
- Python
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- Hugging Face Hub
- Pandas
- NumPy

## 📁 Project Structure
```

├── app.py
├── requirements.txt
├── README.md

````

Model and preprocessing files are downloaded dynamically from Hugging Face Hub.

## 📦 Hugging Face Model Repository
satya-anand-ml/churn-ann-model

Files used:
- ann_churn_model.h5
- onehot_encoder_geo.pkl
- label_encoder_gender.pkl
- scaler.pkl

## 🖥️ Application Features
- Modern gradient-based UI
- Real-time churn probability prediction
- Automatic encoding and scaling
- Clear success and warning messages

## 🧾 Input Features
- Geography
- Gender
- Age
- Credit Score
- Balance
- Tenure
- Number of Products
- Credit Card Status
- Active Member Status
- Estimated Salary

## 📊 Output
- Churn Probability (0–1)
- Prediction Result:
  - Customer is not likely to churn
  - Customer is likely to churn

## ⚙️ Installation & Run Locally

Clone the repository:
```bash
git clone https://github.com/satya-anand-ml/Customer-Churn-Deploy
cd customer-churn-prediction
````

Create virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## 📄 requirements.txt

```txt
streamlit
tensorflow
pandas
numpy
scikit-learn
huggingface-hub
```

## 🧠 Prediction Logic

```python
if churn_probability > 0.5:
    Customer is likely to churn
else:
    Customer is not likely to churn
```

## 🚀 Future Enhancements

* Model explainability (SHAP / LIME)
* Improved UI and animations
* Cloud deployment
* User authentication and prediction history

## 👨‍💻 Author

**Satya Anand**


📧 Email: [satyaanand442@gmail.com](mailto:satyaanand442@gmail.com)
🔗 LinkedIn: https://www.linkedin.com/in/satya-anand-25122003k  
🔗 GitHub: https://github.com/satya2337 

## ⭐ Support

If you like this project, give it a star ⭐ on GitHub and feel free to fork or contribute.

```

