# -*- coding: utf-8 -*-
"""
Created on Thu May 18 05:56:24 2023

@author: evans
"""

import  pickle
import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu
import sklearn
import numpy as np
import pandas as pd

#Loading the saved models
diabetes_model = pickle.load(open('E:\\COURSE DONE\\ML\\M.L Projects\\Multiple Disease Prediction\\saved models\\diabetes_model.sav','rb'),errors='ignore',encoding='latin1')


heart_disease_model = pickle.load(open('E:\\COURSE DONE\\ML\\M.L Projects\\Multiple Disease Prediction\\saved models\\heart_disease_prediction.sav','rb'),errors='ignore',encoding='latin1')
parkinsons_disease_model = pickle.load(open('E:\\COURSE DONE\\ML\\M.L Projects\\Multiple Disease Prediction\\saved models\\Parkinson_disease_prediction.sav','rb'),errors='ignore',encoding='latin1')

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',['Diabetes Prediction',
                                                                 'Heart Disease Prediction',
                                                                 'Parkinsons Disease Prediction',],
                                                                  icons=['activity','heart','person'],
                                                                   default_index=0)

#Diabetes Prediction page
if(selected == 'Diabetes Prediction'):
    #Page title
    st.title('Diabetes Prediction Using ML')
    
    #Input Fields for Variables Column wise
    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Level')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Age = st.text_input('Age of the Patient')
        
        
        
    
    
    #Code for Prediction
    diabetes_diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diabetes_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,
                                                       DiabetesPedigreeFunction,Age]])
        if diabetes_prediction[0] == 1:
            diabetes_diagnosis = 'The Person is Diabetic'
        else:
            diabetes_diagnosis = 'The Person is NOT diabetic'
    st.success(diabetes_diagnosis)
        
elif (selected == 'Heart Disease Prediction'):
    #Page title
    st.title('Heart Disease Prediction Using ML')
    #Input variables column wise
    
    col1,col2,col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Gender')
    with col3:
        cp = st.text_input('Chest Pain Type')
    with col1:
        trestbps = st.text_input('Rest Blood Pressure')
    with col2:
        chol = st.text_input('Cholesterol')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Result')
    with col2:
        thalach = st.text_input('Maximum HeartRate/thalach')
    with col3:
        exang = st.text_input('Exang')
    with col1:
        oldpeak = st.text_input('Old Peak')
    with col2:
        slope = st.text_input('Heart Rate Slope')
    with col3:
        ca = st.text_input("Coranary Artery Calcium")
    with col1:
        thal = st.text_input('Thal')
        
    

    #Code for Prediction
    heart_disease_diagnosis = ''
    if st.button('Heart Disease Prediction'):
        input_data =[[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
        input_data_pd = pd.DataFrame(input_data)
        input_data =input_data_pd.fillna(0)
        input_numeric = pd.to_numeric(list(input_data),errors='coerce')
        input_np = np.array(input_data).reshape(1, -1)
        
        
        heart_disease_pred = heart_disease_model.predict(input_np)
        
        if heart_disease_pred[0] == 1:
            heart_disease_diagnosis ="The Patient's Heart has Defects"
        else:
            heart_disease_diagnosis ="The Patient's Heart is Healthy"
            
    st.success(heart_disease_diagnosis)
else:
    #Page title
    st.title('Parkinsons Disease Prediction using ML')
    
    #Input Variables column wise
    col1,col2,col3= st.columns(3)
    
    with col1:
        MDVP_to_Fo_Ration = st.text_input('MDVP_to_Fo(Hz)_Ration')    
    with col2:
        MDVP_to_Fhi_Ratio = st.text_input('MDVP_to_Fhi(Hz)_Ratio')
    with col3:
        MDVP_to_Flo_Ratio = st.text_input('MDVP_to_Flo(Hz)_Ratio')
    with col1:
        MDVP_jitter_Percentage = st.text_input('MDVP_jitter_Percentage')
    with col2:
         MDVP_to_Jitter_Ratio = st.text_input('MDVP_to_Jitter(Abs)_Ratio')
    with col3:
        MDVP_to_RAP_Ratio = st.text_input('MDVP_to_RAP_Ratio')
    with col1:
        MDVP_to_PPQ_Ratio = st.text_input('MDVP_to_PPQ_Ratio')
    with col2:
        Jitter_to_DDP_Ratio = st.text_input('Jitter_to_DDP_Ratio')
    with col3:
        MDVP_to_Shimmer_Ratio_Abs = st.text_input('MDVP_to_Shimmer_Ratio')
    with col1:
        MDVP_to_Shimmer_dB = st.text_input('MDVP_to_Shimmer(dB)')
    with col2:
        Shimmer_to_APQ3_Ratio = st.text_input('Shimmer_to_APQ3_Ratio')
    with col3:
        Shimmer_to_APQ5_Ratio = st.text_input('Shimmer_to_APQ5_Ratio')
    with col1:
        MDVP_to_APQ_Ratio = st.text_input('MDVP_to_APQ_Ratio')
    with col2:
        Shimmer_to_DDA_Ratio = st.text_input('Shimmer_to_DDA_Ratio')
    with col3:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col1:
        spread1 =st.text_input('spread1')
    with col2:
        spread2 = st.text_input('spread2')
    with col3:
        D2 = st.text_input('D2')
    with col1:
        PPE = st.text_input('PPE')
    
    #code to predict parkinsons disease
    parkinsons_diagnosis = ''
    if st.button('Predict Parkinsons Disease'):
        input = [[MDVP_to_Fo_Ration,MDVP_to_Fhi_Ratio,MDVP_to_Flo_Ratio,MDVP_jitter_Percentage,MDVP_to_Jitter_Ratio,
                  MDVP_to_RAP_Ratio,MDVP_to_PPQ_Ratio,Jitter_to_DDP_Ratio,MDVP_to_Shimmer_Ratio_Abs,MDVP_to_Shimmer_dB,
                  Shimmer_to_APQ3_Ratio,Shimmer_to_APQ5_Ratio,MDVP_to_APQ_Ratio,Shimmer_to_DDA_Ratio,NHR,HNR,RPDE,DFA,spread1,
                  spread2,D2,PPE]]
        parkinsons_prediction = parkinsons_disease_model.predict(input)
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = 'The Patient HAS Parkinsons Disease'
        else:
            parkinsons_diagnosis = 'The Patient DOES NOT have Parkinsons Disease'
    st.success(parkinsons_diagnosis)
            











      
        
    
        
        
    
    

