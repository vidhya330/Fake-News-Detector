📰 Fake News Detector

 Objective

An AI-powered tool to detect whether a news headline is Real or Fake, combining Machine Learning, Natural Language Processing (NLP), and Trusted Source Verification.

 Overview

The Fake News Detector analyzes news headlines and verifies them using trusted sources like BBC, Times of India, CNN, Reuters, and NASA.
It predicts results with confidence, explains how the decision was made, and even lets you upload a full CSV file for batch analysis — all in one interactive Streamlit Dashboard.


Tech Stack

Language: Python

Libraries: scikit-learn, pandas, numpy, joblib, spacy, textblob, lime, plotly, streamlit, feedparser

Model: Logistic Regression

Feature Extraction: TF-IDF Vectorizer

Interface: Streamlit Web App

Project Workflow

1. Data Cleaning – Remove stopwords, punctuation, and convert to lowercase.


2. Vectorization – Transform text into numeric form using TF-IDF.


3. Model Training – Use Logistic Regression to learn from real vs fake news data.


4. Trusted Source Cross-Check – Compare the headline with live feeds from trusted media.


5. Streamlit Dashboard – Visual, easy-to-use interface with analysis and results.

 Project Structure

Fake-News-Detector/
│
├── train_model.py          
├── app.py                  
├── fake_news_model.pkl     
├── vectorizer.pkl          
├── fake_news_dataset.csv   
└── confidence_history.csv  


 How to Run

1. Install dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm


2. Train the model

python train_model.py


3. Run the dashboard

streamlit run app.py

🌟 Key Features

🧠 ML-based Fake News Classification

🔍 Trusted Source Verification (BBC, TOI, CNN, Reuters, NASA)

📈 Confidence Score with Historical Graph

💬 Sentiment & Readability Analysis

🧩 Explainability via LIME (shows why model predicted Fake/Real)

☁ Word Cloud Visualization

📂 Batch CSV Upload Support

🌍 Live News & Trending Topics from Reliable Sources

 Dashboard Tabs Explained
 1. Home

Overview of what the Fake News Detector does.

Quick links to other tabs and short project introduction.

2. Single Headline Analysis

Enter any headline manually.

The model predicts whether it’s Fake, Real, or Uncertain.

Shows confidence percentage, sentiment score, and readability.

Performs live trusted-source verification (checks headline similarity with BBC, TOI, etc.).

Displays a word cloud and highlighted keywords influencing the result.

3. Batch Analysis (CSV Upload)

Upload a .csv file containing multiple headlines.

The system predicts each headline’s label and confidence.

Generates a summary chart of Fake vs Real distribution.

Option to download the analyzed results as a new CSV.

4. Confidence Tracker

Displays past predictions stored in confidence_history.csv.

Visualizes confidence trends over time using line or bar charts.

Helps monitor model stability and accuracy consistency.

5. Trusted News Feed

Fetches live headlines from verified sources like BBC, Reuters, CNN, TOI, and NASA.

Displays trending and factual news updates in real time.

Used for cross-verification and credibility comparison.

 6. Explainability (LIME)

Explains why a specific headline was classified as Fake or Real.

Highlights the words contributing positively or negatively to the prediction.

Makes the model transparent and easy to trust.

7. Word Cloud

Visually represents frequently used words in the dataset or predictions.

Helps identify common fake or real news keywords.

 8. About / Info Tab

Provides project overview, version, author info, and links to documentation.

Useful for understanding the project purpose and developer details.

Result

A smart, explainable, and visually interactive dashboard that detects fake news headlines, cross-verifies them with trusted global sources, and provides users with confidence, insights, and explanations — all in one place.


Developed by

Vidhya 
AI & Data Science Enthusiast
