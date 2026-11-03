import streamlit as st
import joblib
import pandas as pd

LogisticModel = joblib.load('LogisticModelFile.pkl')
x = joblib.load('xValues.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Disease Prediction Web App 💓")
st.markdown("This uses a Logistic Regression model, tested to have 86% accuracy")

Age = st.slider("Age", 0, 100, 25)
Sex = st.selectbox("Gender", ["M", "F"])
ChestPainType = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
RestingBP = st.slider("Resting Blood Pressure", 0, 250, 120)
Cholesterol = st.number_input("Enter Cholestrol", 160)
FastingBS = st.selectbox("Fasting Blood Sugar", ["Yes", "No"])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
MaxHR = st.number_input("MaxHR", 100)
ExerciseAngina = st.selectbox("ExerciseAngina", ["Yes", "No"])
Oldpeak = st.number_input("Oldpeak", 0.0)
ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

btn = st.button("Predict")

if btn:
    user_data = {
        'Age': Age,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': 1 if FastingBS == "Yes" else 0,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': 'Y' if ExerciseAngina == "Yes" else 'N',
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }
    user_df = pd.DataFrame(user_data, index=[0])
    print(user_df.columns)
    strCols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    encoded_df = pd.get_dummies(user_df, columns=strCols, drop_first=False, dtype=int)
    user_df = encoded_df.reindex(columns = x.columns, fill_value = 0)

    scaledData = scaler.transform(user_df)
    scaled_prediction = LogisticModel.predict(scaledData)
    if scaled_prediction[0] == 1:
        st.markdown("<h1 style='text-align: center; color: red;'>The person is likely to have Heart Disease</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: green;'>The person is not likely to have Heart Disease</h1>", unsafe_allow_html=True)


