# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Example dataset
data = {
    'text': [
        'I love this product!',
        'Worst purchase I ever made.',
        'Amazing quality, very happy.',
        'Terrible, do not buy!',
        'I will buy this again.',
        'Not good, very disappointed.',
        'So happy with my purchase!',
        'I hate it.',
        'Best decision ever!',
        'I regret buying this.',
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing: Label encoding
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data into train and test sets
X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into numerical form using TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, 'src/model.pkl')
joblib.dump(vectorizer, 'src/tokenizer.pkl')

print("Model and tokenizer saved successfully!")
