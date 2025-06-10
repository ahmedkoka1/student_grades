import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Student Grade Predictor", layout="centered")
st.title("🎓 Student Final Grade (G3) Predictor")

st.write("Fill in the student information below:")

# Load the trained model pipeline
try:
    model = joblib.load('student_grade.pkl')
except FileNotFoundError:
    st.error("Model file 'student_grade.pkl' not found. Please upload it.")
    st.stop()

# Input fields for the user
school = st.selectbox('School', ['GP', 'MS'])
sex = st.selectbox('Gender', ['F', 'M'])
age = st.slider('Age', 15, 22, 17)
address = st.selectbox('Address', ['U', 'R'])
famsize = st.selectbox('Family Size', ['LE3', 'GT3'])
Pstatus = st.selectbox('Parental Status', ['T', 'A'])
Medu = st.slider("Mother's Education", 0, 4, 2)
Fedu = st.slider("Father's Education", 0, 4, 2)
Mjob = st.selectbox("Mother's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
Fjob = st.selectbox("Father's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
reason = st.selectbox("Reason for Choosing School", ['home', 'reputation', 'course', 'other'])
guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
traveltime = st.slider("Travel Time", 1, 4, 1)
studytime = st.slider("Study Time", 1, 4, 2)
failures = st.slider("Past Class Failures", 0, 3, 0)
schoolsup = st.selectbox("School Support", ['yes', 'no'])
famsup = st.selectbox("Family Support", ['yes', 'no'])
paid = st.selectbox("Extra Paid Classes", ['yes', 'no'])
activities = st.selectbox("Extracurricular Activities", ['yes', 'no'])
nursery = st.selectbox("Attended Nursery School", ['yes', 'no'])
higher = st.selectbox("Wants Higher Education", ['yes', 'no'])
internet = st.selectbox("Internet Access", ['yes', 'no'])
romantic = st.selectbox("Romantic Relationship", ['yes', 'no'])
famrel = st.slider("Family Relationship Quality", 1, 5, 4)
freetime = st.slider("Free Time", 1, 5, 3)
goout = st.slider("Going Out", 1, 5, 3)
Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)
health = st.slider("Health Status", 1, 5, 3)
absences = st.slider("Number of Absences", 0, 93, 4)
G1 = st.slider("Grade Period 1 (G1)", 0, 20, 10)
G2 = st.slider("Grade Period 2 (G2)", 0, 20, 10)

# Derived Features
log_absences = np.log1p(absences)
g1_absences = G1 + log_absences
g1_health = G1 + health
Medu_fedu = Medu + Fedu
final_degree = G1 + G2

# Construct input data
input_data = pd.DataFrame([{
    'school': school,
    'sex': sex,
    'age': age,
    'address': address,
    'famsize': famsize,
    'Pstatus': Pstatus,
    'Medu': Medu,
    'Fedu': Fedu,
    'Mjob': Mjob,
    'Fjob': Fjob,
    'reason': reason,
    'guardian': guardian,
    'traveltime': traveltime,
    'studytime': studytime,
    'failures': failures,
    'schoolsup': schoolsup,
    'famsup': famsup,
    'paid': paid,
    'activities': activities,
    'nursery': nursery,
    'higher': higher,
    'internet': internet,
    'romantic': romantic,
    'famrel': famrel,
    'freetime': freetime,
    'goout': goout,
    'Dalc': Dalc,
    'Walc': Walc,
    'health': health,
    'absences': log_absences,
    'G1': G1,
    'G2': G2,
    'g1_absences': g1_absences,
    'g1_health': g1_health,
    'Medu_fedu': Medu_fedu,
    'final_degree': final_degree
}])

# Predict G3 when the button is clicked
if st.button("🎯 Predict Final Grade (G3)"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Final Grade (G3): {round(prediction[0], 2)}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
