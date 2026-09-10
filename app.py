import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("Breast Cancer Outcome Predictor")
st.write("Trained on the Breast Cancer Wisconsin dataset. For educational purposes only.")

@st.cache_resource
def load_model_and_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X, data.target_names

model, X_train, target_names = load_model_and_data()

st.sidebar.header("Patient information")

def user_input_features():
    radius_mean = st.sidebar.number_input('Radius Mean', float(X_train['mean radius'].min()), float(X_train['mean radius'].max()), float(X_train['mean radius'].mean()))
    texture_mean = st.sidebar.number_input('Texture Mean', float(X_train['mean texture'].min()), float(X_train['mean texture'].max()), float(X_train['mean texture'].mean()))
    perimeter_mean = st.sidebar.number_input('Perimeter Mean', float(X_train['mean perimeter'].min()), float(X_train['mean perimeter'].max()), float(X_train['mean perimeter'].mean()))
    area_mean = st.sidebar.number_input('Area Mean', float(X_train['mean area'].min()), float(X_train['mean area'].max()), float(X_train['mean area'].mean()))
    smoothness_mean = st.sidebar.number_input('Smoothness Mean', float(X_train['mean smoothness'].min()), float(X_train['mean smoothness'].max()), float(X_train['mean smoothness'].mean()))
    
    user_data = {
        'mean radius': radius_mean,
        'mean texture': texture_mean,
        'mean perimeter': perimeter_mean,
        'mean area': area_mean,
        'mean smoothness': smoothness_mean
    }
    
    features_df = pd.DataFrame(columns=X_train.columns)
    features_df.loc[0] = X_train.mean()
    
    for key, value in user_data.items():
        features_df.loc[0, key] = value
        
    return features_df

df = user_input_features()

st.subheader("User Input Parameters")
st.write(df)

if st.button("Predict"):
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    st.subheader("Prediction Result")
    res_text = target_names[prediction[0]]
    
    if prediction[0] == 0:
        st.error(f"The model predicts: **Malignant** (1)")
    else:
        st.success(f"The model predicts: **Benign** (0)")
        
    st.subheader("Prediction Probability")
    st.write(prediction_proba)
