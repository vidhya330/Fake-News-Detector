# model.py
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model & vectorizer
with open("fake_news_model.pkl", "rb") as f:
    ml_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Predict function
def analyze_headline_ml(headline, confidence_threshold=0.75):
    vector = vectorizer.transform([headline])
    pred_prob = ml_model.predict_proba(vector)[0][1]

    if pred_prob >= confidence_threshold:
        label = "Real"
    elif pred_prob <= (1 - confidence_threshold):
        label = "Fake"
    else:
        label = "Uncertain / Needs Fact-check"

    return label,pred_prob*100
