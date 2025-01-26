# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the model and tokenizer (TF-IDF vectorizer)
model = joblib.load('src/model.pkl')
tokenizer = joblib.load('src/tokenizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Sentiment Analysis API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    
    # Transform the text using the tokenizer
    text_tfidf = tokenizer.transform([text])
    
    # Predict sentiment
    prediction = model.predict(text_tfidf)
    
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    
    return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    app.run(debug=True)
