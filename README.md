# Disease-Prediction-Model
Built model that will predict whether a person have a disease or not.

# ⚕️ Multi-Disease Prediction System

An end-to-end Machine Learning web application built with **Python**, **Scikit-Learn**, and **Streamlit** to predict the risk of **Diabetes**, **Heart Disease**, and **Parkinson’s Disease** based on clinical parameters.

## 📌 Overview
This project serves as a **Clinical Decision Support System (CDSS)** prototype. It takes standard medical metrics and lab results, applies feature standardization, and runs them through pre-trained Support Vector Machine (SVM) models to provide instant risk classifications through a clean, interactive web interface.

## 🩺 Diseases & Datasets Covered
1. **Diabetes Prediction:** Trained on the Pima Indians Diabetes Dataset (8 features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age).
2. **Heart Disease Prediction:** Trained on the Heart Disease Dataset (13 features: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, EKG Results, Max HR, Exercise Angina, ST Depression, Slope, Major Vessels, Thallium).
3. **Parkinson's Disease Prediction:** Trained on the Oxford Parkinson's Disease Detection Dataset (22 biomedical voice measurements including MDVP, Jitter, Shimmer, NHR, HNR, RPDE, DFA, and PPE).

## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Machine Learning:** Scikit-Learn (Support Vector Machine - Linear Kernel, `StandardScaler`)
* **Data Manipulation:** Pandas, NumPy
* **Model Serialization:** Pickle (`.sav` files)
* **Web Framework:** Streamlit

## 📂 Project Structure
```text
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies for deployment
├── diabetes_model.sav         # Trained SVM model for Diabetes
├── heart_model.sav            # Trained SVM model for Heart Disease
├── parkinsons_model.sav       # Trained SVM model for Parkinson's
├── scaler_diabetes.sav        # StandardScaler fitted on Diabetes data
├── scaler_heart.sav           # StandardScaler fitted on Heart Disease data
├── scaler_parkinsons.sav      # StandardScaler fitted on Parkinson's data
└── README.md                  # Project documentation
