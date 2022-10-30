import xgboost as xgb
import streamlit as st
import pandas as pd

#Loading up the Regression model we created
model = xgb.XGBClassifier()
model.load_model('xgb_model.json')

#Caching the model for faster loading
@st.cache

def predict(m_bb,m_wwbb,jet_1_pt,m_wbb,m_jjj):
    prediction = model.predict(pd.DataFrame([[m_bb,m_wwbb,jet_1_pt,m_wbb,m_jjj]], columns=['m_bb','m_wwbb','jet 1 pt','m_wbb','m_jjj']))
    return prediction

st.title('HIGGS BOSON Predictor')
st.header('Enter the characteristics of the particle:')

m_bb = st.number_input('m_bb:', min_value=0.001, max_value=10.0, value=1.0,key="1")

m_wwbb = st.number_input('m_wwbb:', min_value=0.001, max_value=10.0, value=1.0,key="2")

jet_1_pt = st.number_input('jet_1_pt', min_value=0.001, max_value=10.0, value=1.0,key="3")

m_wbb = st.number_input('m_wbb', min_value=0.001, max_value=10.0, value=1.0,key="4")

m_jjj = st.number_input('m_jjj', min_value=0.001, max_value=10.0, value=1.0,key="5")


if st.button('Predict '):

    label = predict(m_bb,m_wwbb,jet_1_pt,m_wbb,m_jjj)
    st.success(f'The predicte value is {label[0]:.2f}')

