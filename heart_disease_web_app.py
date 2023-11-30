# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:16:46 2023

@author: Rahul Thakur
"""

import numpy as np
import pickle
import streamlit as st
#loading the saved model
loaded_model=pickle.load(open('C:/Users/Rahul Thakur/Desktop/Mini Project/trained_model_svm.sav', 'rb'))

def heart_disease_prediction(input_data):
    
    input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]== 0):
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'



def main():
    
    #creating a title 
    st.title('Heart Disease prediction Web APP')
    
    #getting the input from the users
    
    age=st.text_input('Enter Your Age')
    sex=st.text_input('Enter Sex: 0 for male , 1 for female')
    cp=st.text_input('Enter Chest pain type 0-3')
    trestbps=st.text_input('Enter Resting blood pressure or bp between 90-200')
    chol=st.text_input('Enter Your cholestrol between 100-400')
    fbs=st.text_input('Enter Your fasting blood pressure from 0 to 1')
    restecg=st.text_input('Enter Your electrocardiographic result from 0 to 1')
    thalach=st.text_input('Enter Your Maximum heart rate between 90-200')
    exang=st.text_input('Enter Your exercise induced Angina from 0-1')
    oldpeak=st.text_input('Enter Your ST depression induced by exercise relative to rest from 0-5.0')
    slope=st.text_input('Enter the slope of the peak exercise ST segment from 0-2')
    ca=st.text_input('Enter  number of major vessels (0-3) colored by fluoroscopy from 0-4')
    thal=st.text_input('Enter  [normal; fixed defect; reversible defect from 1-3]')
    
    #code for prediction
    diagnosis=''
    
    #creating a button for prediction
    if st.button('Heart Disease Prediction Result'):
         diagnosis=heart_disease_prediction([age,sex,cp ,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
         
    st.success(diagnosis)


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    