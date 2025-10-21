
import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import feedparser
import spacy
from wordcloud import WordCloud
from textblob import TextBlob
from textstat import flesch_reading_ease
from scipy.special import expit
from lime.lime_text import LimeTextExplainer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

ml_model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"
]
TRUSTED_SOURCES = ["bbc.co.uk", "reuters.com", "timesofindia.indiatimes.com", "nasa.gov", "cnn.com"]

CONF_HISTORY = "confidence_history.csv"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text.strip()

def ml_predict_proba(texts):
    vectors = vectorizer.transform(texts)
    scores = ml_model.decision_function(vectors)
    probs_real = expit(scores)
    probs_fake = 1 - probs_real
    return np.column_stack([probs_fake, probs_real])

@st.cache_data(ttl=600)
def fetch_live_news_rss(limit=10):
    headlines = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit]:
            headlines.append(entry.title)
    return headlines if headlines else ["No live news available."]

def get_sentiment(text): return TextBlob(text).sentiment.polarity
def get_readability(text): return max(0, min(flesch_reading_ease(text), 100))

def generate_wordcloud(text):
    if not text.strip(): return None
    wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(text)
    return wc.to_image()

def get_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def source_credibility(url):
    if any(src in url for src in TRUSTED_SOURCES):
        return "✅ Trusted"
    elif "blog" in url or "medium.com" in url or "wordpress" in url:
        return "⚠ Suspicious"
    else:
        return "❓ Unknown"

def get_trending_topics(headlines):
    words = " ".join(headlines)
    doc = nlp(words)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON"]]
    if not entities:
        return pd.DataFrame(columns=["Entity", "Count"])
    df = pd.DataFrame(entities, columns=["Entity"])
    top = df.value_counts().head(10).reset_index()
    top.columns = ["Entity", "Count"]
    return top

def explain_simple(label, confidence):
    if label == "Real":
        return f"This news looks *real* because it matches patterns in reliable sources. Confidence: {confidence:.1f}%"
    else:
        return f"This looks *fake* because it may use emotional or exaggerated words, or it wasn’t found in trusted sources. Confidence: {confidence:.1f}%"

def fact_check_online(headline):
    found_sources = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            try:
                if headline.lower() in entry.title.lower():
                    found_sources.append(entry.link)
            except Exception:
                continue
    trusted_links = [link for link in found_sources if any(src in link for src in TRUSTED_SOURCES)]
    return len(trusted_links) > 0, trusted_links

def analyze_headline_ml_online(headline):
    cleaned = preprocess_text(headline)
    vector = vectorizer.transform([cleaned])
    score = ml_model.decision_function(vector)[0]
    pred_prob = expit(score)
    threshold = 0.6
    ml_label = "Real" if pred_prob >= threshold else "Fake"

    is_verified, trusted_links = fact_check_online(headline)
    final_label = "Real" if is_verified else ml_label
    final_confidence = max(pred_prob, 0.7) * 100 if is_verified else pred_prob * 100

    sentiment = get_sentiment(headline)
    readability = get_readability(headline)
    lime_words, exp = get_lime_explanation(cleaned)

    explanation_text = f"ML model predicts {ml_label} with {pred_prob*100:.2f}% confidence. "
    if is_verified:
        explanation_text += f"Verified in trusted sources: {', '.join(trusted_links)}. "
    explanation_text += f"Influential words: {lime_words}"

    save_confidence_timeline(headline, final_label, final_confidence)

    return {
        "headline": headline,
        "label": final_label,
        "confidence": final_confidence,
        "sentiment": sentiment,
        "readability": readability,
        "explanation": explanation_text,
        "lime_exp": exp,
        "trusted_links": trusted_links
    }

def get_lime_explanation(headline):
    exp = explainer.explain_instance(headline, ml_predict_proba, num_features=5)
    lime_exp = dict(exp.as_list())
    explanation_words = ", ".join([f"{k}: {v:.2f}" for k, v in lime_exp.items()])
    return explanation_words, exp

def save_confidence_timeline(headline, label, confidence):
    data = pd.DataFrame([[time.strftime("%H:%M:%S"), headline[:40], label, confidence]],
                        columns=["Time", "Headline", "Label", "Confidence"])
    if os.path.exists(CONF_HISTORY):
        old = pd.read_csv(CONF_HISTORY)
        df = pd.concat([old, data]).tail(30)
    else:
        df = data
    df.to_csv(CONF_HISTORY, index=False)

def analyze_batch_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if "headline" not in df.columns:
            st.error("CSV must contain a 'headline' column.")
            return None
        results = []
        for idx, row in df.iterrows():
            res = analyze_headline_ml_online(str(row['headline']))
            results.append({
                "headline": res['headline'],
                "label": res['label'],
                "confidence": res['confidence'],
                "explanation": explain_simple(res['label'], res['confidence'])
            })
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

st.set_page_config(page_title="Fake News Dashboard", page_icon="📰", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#fad0c4,#ffd1ff);
    font-family: 'Poppins', sans-serif;
}
.main-card {
    background: rgba(255,255,255,0.95);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 25px rgba(0,0,0,0.2);
    margin-bottom: 25px;
}
.stButton button {
    background-color:#ff4b4b!important;
    color:white!important;
    border-radius:10px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(90deg,#ff4b4b,#ff6f91);padding:20px;border-radius:15px;text-align:center;">
<h1 style="color:white;">📰 Fake News Intelligence Dashboard</h1>
<p style="color:white;font-size:18px;">AI + NLP + Batch Analysis + Real-Time Fact Checking + Visualization</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🧠 Analyze Headline", "📂 Batch Analysis", "📈 Confidence Timeline", "🌍 Trending Topics", "📰 Live News", "ℹ About"])

with tabs[0]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("Analyze a Headline")
    headline_input = st.text_area("Enter news headline:", height=120)
    if st.button("🔍 Analyze Headline"):
        if headline_input.strip():
            ml_result = analyze_headline_ml_online(headline_input)
            color = "#28a745" if ml_result['label']=="Real" else "#dc3545"
            st.markdown(f"*Prediction:* <span style='color:{color};'>{ml_result['label']}</span> ({ml_result['confidence']:.2f}%)", unsafe_allow_html=True)
            st.markdown("### 🧩 Named Entities")
            entities = get_entities(headline_input)
            if entities:
                for ent, label in entities:
                    st.markdown(f"<span style='background:#ffe5ec;padding:5px;border-radius:8px;'>{ent}</span> <i>({label})</i>", unsafe_allow_html=True)
            wc = generate_wordcloud(headline_input)
            if wc: st.image(wc, caption="WordCloud", use_column_width=True)

            labels = ["Confidence", "Sentiment", "Readability"]
            values = [ml_result['confidence']/100, abs(ml_result['sentiment']), ml_result['readability']/100]
            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill='toself',
                line_color=color
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10),
                width=350,
                height=350
            )
            st.plotly_chart(fig, use_container_width=False)

            st.subheader("🪄 Explain Like I’m 5")
            st.write(explain_simple(ml_result['label'], ml_result['confidence']))
            st.subheader("💬 Full Explanation")
            st.info(ml_result["explanation"])
    st.markdown('</div>', unsafe_allow_html=True)
with tabs[1]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("📂 Batch CSV Analysis")
    uploaded_file = st.file_uploader("Upload CSV with 'headline' column", type=["csv"])
    if uploaded_file:
        batch_results = analyze_batch_csv(uploaded_file)
        if batch_results is not None:
            st.success("Batch analysis completed!")
            st.dataframe(batch_results)
            csv = batch_results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, "batch_results.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)
with tabs[2]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("📈 Confidence Timeline")
    if os.path.exists(CONF_HISTORY):
        df_hist = pd.read_csv(CONF_HISTORY)
        fig = go.Figure(data=go.Scatter(x=df_hist["Time"], y=df_hist["Confidence"], mode='lines+markers', line=dict(color='#ff4b4b')))
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), width=700, height=400)
        st.plotly_chart(fig, use_container_width=False)
        st.dataframe(df_hist)
    else:
        st.info("No confidence history yet. Analyze some headlines first!")
    st.markdown('</div>', unsafe_allow_html=True)
with tabs[3]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("🌍 Trending Topics (BBC + TOI)")
    news = fetch_live_news_rss(limit=20)
    top = get_trending_topics(news)
    if not top.empty:
        st.bar_chart(top.set_index("Entity")["Count"])
    else:
        st.info("No trending topics found.")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[4]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("📰 Live News Headlines")
    news = fetch_live_news_rss(limit=15)
    for n in news:
        st.markdown(f"- {n}")
    st.markdown('</div>', unsafe_allow_html=True)

with st.tabs(["ℹ About"])[0]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("ℹ About")
    st.markdown("""
    This dashboard combines *Machine Learning, NLP, Batch CSV Analysis, and Real-Time Fact Checking*  
    to intelligently detect fake news and visualize its credibility.

    *Features:*
    - 🧠 ML + LIME + Explainability
    - 🧩 Named Entity Recognition
    - 💬 Simple Human-Friendly Explanations
    - 📂 Batch CSV Analysis
    - 🌍 Trending Topics Visualization
    - 📰 Live News Headlines
    - 📈 Confidence History Timeline
*Developed by:* Vidhya 🌸
                """)
    st.markdown('</div',
                unsafe_allow_html=True)

