import streamlit as st
import pickle

# Load trained model & vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Apply Custom CSS
st.markdown("""
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(to right, #2c5d6a, #265a99);
            background-size: cover;
            color: white;
            text-align: center;
            padding: 40px;
        }

        .container {
             width: 800px; 
            max-width: 90%;
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }

        .description {
            font-size: 18px;
            margin-bottom: 20px;
        }

        textarea {
            width: 90%;
            height: 100px;
            padding: 10px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
            outline: none;
            margin-bottom: 20px;
        }

        button {
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 6px;
            border: none;
            background-color: #ffcc00;
            color: black;
            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            background-color: #ffaa00;
        }

        .result-box {
            margin-top: 20px;
            font-size: 22px;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
        }

        .positive {
            color: #3cf053;
        }

        .negative {
            color: #ff4f4f;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Input Box
st.title("💬 Sentiment Analysis App")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    text_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(text_vectorized)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 🙁"
    st.markdown(f"<h2 class='{sentiment.lower()}'>{sentiment}</h2>", unsafe_allow_html=True)
