import pickle
import streamlit as st
import numpy as np

# Configure the main webpage
st.set_page_config(page_title="Disease Prediction System", layout="wide", page_icon="⚕️")

# Load the saved models and scalers (reading in binary mode 'rb')
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

scaler_diabetes = pickle.load(open('scaler_diabetes.sav', 'rb'))
scaler_heart = pickle.load(open('scaler_heart.sav', 'rb'))
scaler_parkinsons = pickle.load(open('scaler_parkinsons.sav', 'rb'))

# Build the sidebar menu
with st.sidebar:
    st.title("Medical Assistant")
    selected = st.selectbox("Choose a Disease to Predict", 
                            ["Diabetes Prediction", 
                             "Heart Disease Prediction", 
                             "Parkinson's Prediction"])

# ==========================================
# DIABETES PREDICTION PAGE
# ==========================================
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction System")
    
    # Create three columns for a clean, organized layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', value='0')
    with col2:
        Glucose = st.text_input('Glucose Level', value='0')
    with col3:
        BloodPressure = st.text_input('Blood Pressure', value='0')
    with col1:
        SkinThickness = st.text_input('Skin Thickness', value='0')
    with col2:
        Insulin = st.text_input('Insulin Level', value='0')
    with col3:
        BMI = st.text_input('BMI', value='0')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', value='0')
    with col2:
        Age = st.text_input('Age', value='0')
        
    diab_diagnosis = ''
    
    # Create the prediction button
    if st.button('Get Diabetes Test Result'):
        # 1. Convert the text inputs into numbers (floats)
        user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), 
                      float(SkinThickness), float(Insulin), float(BMI), 
                      float(DiabetesPedigreeFunction), float(Age)]
        
        # 2. Scale the input data using the exact same scaler from training
        scaled_input = scaler_diabetes.transform([user_input])
        
        # 3. Make the prediction
        diab_prediction = diabetes_model.predict(scaled_input)
        
        # 4. Display the result
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The patient is classified as Diabetic.'
        else:
            diab_diagnosis = 'The patient is classified as Non-Diabetic.'
            
    st.success(diab_diagnosis)
    # ==========================================
# HEART DISEASE PREDICTION PAGE
# ==========================================
elif selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction System")
    
    # 13 input features based on your heart.csv dataset
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age', value='0')
    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)', value='0')
    with col3:
        cp = st.text_input('Chest Pain Type (0-3)', value='0')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', value='0')
    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl', value='0')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 (1=True, 0=False)', value='0')
    with col1:
        restecg = st.text_input('Resting EKG Results (0-2)', value='0')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved', value='0')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)', value='0')
    with col1:
        oldpeak = st.text_input('ST Depression', value='0')
    with col2:
        slope = st.text_input('Slope of ST Segment (0-2)', value='0')
    with col3:
        ca = st.text_input('Number of Major Vessels (0-4)', value='0')
    with col1:
        thal = st.text_input('Thallium Test Result (0-3)', value='0')
        
    heart_diagnosis = ''
    
    if st.button('Get Heart Disease Test Result'):
        # 1. Convert the text inputs into numbers (floats)
        user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), 
                      float(fbs), float(restecg), float(thalach), float(exang), 
                      float(oldpeak), float(slope), float(ca), float(thal)]
        
        # 2. Scale the input data
        scaled_input = scaler_heart.transform([user_input])
        
        # 3. Make the prediction
        heart_prediction = heart_model.predict(scaled_input)
        
        # 4. Display the result
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The patient is classified as having Heart Disease.'
        else:
            heart_diagnosis = 'The patient is classified as having a Healthy Heart.'
            
    st.success(heart_diagnosis)
    # ==========================================
# PARKINSON'S PREDICTION PAGE
# ==========================================
elif selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction System")
    
    # 22 input features structured into 4 columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', value='0')
        jitter_abs = st.text_input('MDVP:Jitter(Abs)', value='0')
        shimmer = st.text_input('MDVP:Shimmer', value='0')
        apq = st.text_input('MDVP:APQ', value='0')
        rpde = st.text_input('RPDE', value='0')
        d2 = st.text_input('D2', value='0')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)', value='0')
        rap = st.text_input('MDVP:RAP', value='0')
        shimmer_db = st.text_input('MDVP:Shimmer(dB)', value='0')
        dda = st.text_input('Shimmer:DDA', value='0')
        dfa = st.text_input('DFA', value='0')
        ppe = st.text_input('PPE', value='0')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)', value='0')
        ppq = st.text_input('MDVP:PPQ', value='0')
        apq3 = st.text_input('Shimmer:APQ3', value='0')
        nhr = st.text_input('NHR', value='0')
        spread1 = st.text_input('spread1', value='0')
        
    with col4:
        jitter_percent = st.text_input('MDVP:Jitter(%)', value='0')
        ddp = st.text_input('Jitter:DDP', value='0')
        apq5 = st.text_input('Shimmer:APQ5', value='0')
        hnr = st.text_input('HNR', value='0')
        spread2 = st.text_input('spread2', value='0')
        
    parkinsons_diagnosis = ''
    
    if st.button("Get Parkinson's Test Result"):
        # 1. Convert all 22 text inputs into numbers (floats) in the exact order the model expects
        user_input = [float(fo), float(fhi), float(flo), float(jitter_percent), float(jitter_abs), 
                      float(rap), float(ppq), float(ddp), float(shimmer), float(shimmer_db), 
                      float(apq3), float(apq5), float(apq), float(dda), float(nhr), float(hnr), 
                      float(rpde), float(dfa), float(spread1), float(spread2), float(d2), float(ppe)]
        
        # 2. Scale the input data
        scaled_input = scaler_parkinsons.transform([user_input])
        
        # 3. Make the prediction
        parkinsons_prediction = parkinsons_model.predict(scaled_input)
        
        # 4. Display the result
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The patient is classified as having Parkinson's Disease."
        else:
            parkinsons_diagnosis = "The patient is classified as having NO Parkinson's Disease."
            
    st.success(parkinsons_diagnosis)
