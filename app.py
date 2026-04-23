import streamlit as st
import pickle

st.title("IPL Match Winner Prediction")

model = pickle.load(open("model.pkl", "rb"))

team1 = st.number_input("Team 1 (encoded value)")
team2 = st.number_input("Team 2 (encoded value)")

if st.button("Predict"):
    result = model.predict([[team1, team2]])
    st.write("Predicted Winner:", result)