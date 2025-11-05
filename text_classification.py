import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
   
   
   
st.markdown("""
        <div style="text-align: center; color: #e74c3c; font-weight: bold; margin-top: 50px;">
            📝 Partagez un tweet pour que nous puissions l’analyser.
        </div>
    """, unsafe_allow_html=True)

# Define constants
max_words = 10000
max_len = 100

sentiment_labels = ['Negative', 'Neutral', 'Positive', 'Irrelevant']

# Load model and tokenizer 
@st.cache_resource
def load_sentiment_model():
    model = load_model("sentiment_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        return model, tokenizer


model, tokenizer = load_sentiment_model()

   

# User input
user_text = st.text_area("Enter a sentence to analyze its sentiment:")

if st.button("Analyze Sentiment"):
    if user_text.strip():
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_text])
        pad = pad_sequences(seq, maxlen=max_len)

        # Predict
        pred = model.predict(pad)
        sentiment_idx = np.argmax(pred)
        sentiment = sentiment_labels[sentiment_idx]
        confidence = pred[0][sentiment_idx]

            # Display result
        st.success(f"**Predicted Sentiment:** {sentiment} ({confidence:.2%} confidence)")
    else:
        st.warning("Please enter some text before analyzing.")