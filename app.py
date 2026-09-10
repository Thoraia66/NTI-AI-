import streamlit as st
st.title("My Streamlit App")
st.header("Welcome to my app!")
st.number_input("Age", min_value=0, max_value=100, value=50)
st.slider("Mean radius", min_value=0, max_value=100, value=50)
st.selectbox("Sex", ["male", "female"])
if st.button("predict"):
    st.success("Prediction successful!")
