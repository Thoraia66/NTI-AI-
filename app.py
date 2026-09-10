import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("Breast Cancer Outcome Predictor")
st.write("Trained on the Breast Cancer Wisconsin dataset. For educational purposes.")

@st.cache_resource
def load_model_and_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

X, y = load_model_and_data()
st.write("Data loaded successfully!")
