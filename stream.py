import xgboost as xgb
import streamlit as st
import pandas as pd
import joblib


#Loading up the Regression model we created
model = xgb.XGBClassifier()
model.load_model('xgb_model.json')

# model = joblib.load("model.pkl")

#Caching the model for faster loading
@st.cache

def predict(m_bb,m_wwbb,jet_1_pt,m_wbb,m_jjj):
    prediction = model.predict(pd.DataFrame([[m_bb,m_wwbb,jet_1_pt,m_wbb,m_jjj]], columns=['m_bb','m_wwbb','jet 1 pt','m_wbb','m_jjj']))
    return prediction

st.title('HIGGS BOSON Predictor')
st.header('Enter the characteristics of the particle:')
st.text("The Higgs boson is the fundamental particle associated with the Higgs field, a field that gives mass to other fundamental particles such as electrons and quarks. A particle's mass determines how much it resists changing its speed or position when it encounters a force. Not all fundamental particles have mass.")

m_bb = st.number_input('m_bb:', min_value=0.001, max_value=10.0, value=1.0,key="1",format="%.4f")

m_wwbb = st.number_input('m_wwbb:', min_value=0.001, max_value=10.0, value=1.0,key="2",format="%.4f")

jet_1_pt = st.number_input('jet_1_pt: The transverse momentum of the leading jet, that is the jet with largest transverse momentum', min_value=0.001, max_value=10.0, value=1.0,key="3",format="%.4f")

m_wbb = st.number_input('m_wbb', min_value=0.001, max_value=10.0, value=1.0,key="4",format="%.4f")

m_jjj = st.number_input('m_jjj', min_value=0.001, max_value=10.0, value=1.0,key="5",format="%.4f")


if st.button('Predict '):

    label = predict(m_bb,m_wwbb,jet_1_pt,m_wbb,m_jjj)
    if label[0]>0:
        st.success(f'The predicte value is: High Energy Particle')
    else:
        st.success(f'The predicte value is: Not a High Energy Particle')

