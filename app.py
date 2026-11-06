import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

API_URL="http://127.0.0.1:8000/predict"
# set configuration

st.set_page_config(
        page_title="Diabetes Prediction App",
        page_icon=" .",
        layout="centered"
)

st.title(" - Diabetes Prediction App ")
st.write("Entre patient and click")

st.subheader("Patient Details")

col1,col2 =st.columns(2)

with col1:
    Pregnancies=st.number_input("Pregnancies",0,20,2)
    Glucose = st.number_input("Glucose", 0, 300, 120)
    BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
    
with col2:
    Insulin=st.number_input("Insulin",0,900,80)
    BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
    Dpf = st.number_input(
        "Diabetes Pedigree Function", 0.0, 3.0, 0.5
    )
    Age = st.number_input("Age", 1, 120, 30)
    
if st.button("predict"):
    diabetes_input = {
                    "Pregnancies":Pregnancies ,
                    "Glucose": Glucose,
                    "BloodPressure": BloodPressure,
                    "SkinThickness": SkinThickness,
                    "Insulin": Insulin,
                    "BMI": BMI,
                    "DiabetesPedigreeFunction": Dpf,
                    "Age": Age
                }
    reponse=requests.post(API_URL,json=diabetes_input)
    rep_dict=reponse.json()
    prediction=rep_dict['Prediction']
    


    st.divider()
    if prediction == 1:
        st.error("Model predicted Diabtes **(+ve)**")
    else:
        st.success("Model predicted  no Diabtes **(-ve)**")
    