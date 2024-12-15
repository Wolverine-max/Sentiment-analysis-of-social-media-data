import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS  
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np


st.title('Sentiment Analysis App')

st.write('This app is for sentiment analysis of social media data !')
st.sidebar.title('Sentiment analysis of Social Media Data')
with st.expander('Twitter Data'):
  st.write('**Raw data**')
  df1=pd.read_csv('https://raw.githubusercontent.com/Wolverine-max/Sentiment-analysis-of-social-media-data/refs/heads/master/Twitter_Data.csv')
  df1.columns=['text','labels']
  st.dataframe(df1)
with st.expander('Reddit Data'):
  st.write('**Raw data**')
  df2=pd.read_csv('https://raw.githubusercontent.com/Wolverine-max/Sentiment-analysis-of-social-media-data/refs/heads/master/Reddit_Data.csv')
  df2.columns=['text','labels']
  st.dataframe(df2)
with st.expander('Merged Data'):
  st.write("**Merged Data**")
  data = pd.concat([df1, df2], ignore_index=True)  # Concatenating the datasets
  st.dataframe(data) 
  
model = pickle.load(open('logreg.pkl','rb')) 
vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb')) 

def predict_sentiment(text):
    text_vector = vectorizer.transform([text])  # Transform the text to the vector
    sentiment = model.predict(text_vector)[0]  # Predict sentiment
    return sentiment

st.title('Sentiment Analysis Using Pre-trained Model')

# Text input for analysis
input_text = st.text_area('Enter the text to analyze sentiment ( tweet or Reddit post)')

# Show results when button is clicked
if st.button('Analyze Sentiment') and input_text:
    sentiment = predict_sentiment(input_text)
    
    # Display sentiment
    st.write(f'Sentiment: {sentiment}')
    if sentiment > 0:
        st.write("Sentiment: Positive")
    elif sentiment < 0:
        st.write("Sentiment: Negative")
    else:
        st.write("Sentiment: Neutral")


# Background Image (optional)
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1620794511798-d7ba5299a087?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mzh8fHNvY2lhbCUyMG1lZGlhfGVufDB8fDB8fHww");
            background-size: cover;
        }
    </style>
""", unsafe_allow_html=True)
