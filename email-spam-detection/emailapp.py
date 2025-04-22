import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
with open('vectorizer.pkl.url', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl.url', 'rb') as f:
    model = pickle.load(f)

# Set Streamlit theme and page configuration
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

# Custom CSS for styling and animations
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f4f4;
        color: #444;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        transition: transform 0.2s;
    }
    .stButton button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stTextArea textarea {
        background-color: #fff5e6;
        border-radius: 50px;
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    .stTextArea textarea:hover {
    background-color: #fff5e6;
    border-radius: 50px;
    animation: glow 1.5s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from {
            box-shadow: 0 0 50px #ff9900;
        }
        to {
            box-shadow: 0 0 50px #ffcc00;
        }
    }
    .stTitle {
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stSuccess {
        color: #28a745;
        font-weight: bold;
        text-align: center;
        animation: fadein 2s;
    }
    @keyframes fadein {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# UI
st.title("📧 Email Spam Detection App")
st.write("Enter an email message below to check if it's spam or not. 🚫📬")

# Input text box
user_input = st.text_area("Email Message", placeholder="Paste your email content here...", height=200)

if st.button("Predict"):
    input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(input_tfidf)[0]
    result = "This is a Spam Message" if prediction == 1 else "This is Not a Spam Message"
    st.success(f"Prediction: {result}")
