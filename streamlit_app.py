import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
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
  
lm = WordNetLemmatizer()

# Replacing URL with 'URL'
def replace_url(text):
    return re.sub(r'https?:\/\/\S*|www\.\S+', 'URL', text)

# Removing HTML tags
def remove_html(text):
    return re.sub(r'<.*?>', '', text)

# Replacing mentions with 'user'
def replace_mentions(text):
    return re.sub(r'@\S+', 'user', text, flags=re.IGNORECASE)

# Replacing numbers with 'NUMBER'
def replace_num(text):
    return re.sub(r'\b\d+\b', 'NUMBER', text)

# Replacing <3 with 'HEART'
def replace_heart(text):
    return re.sub(r'<3', 'HEART', text)

# Removing alphanumeric characters (e.g., XYZ123ABC)
def remove_alphanumeric(text):
    return re.sub(r'\w*\d+\w*', '', text)

# Removing all English stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Removing punctuation
def remove_punctuations(text):
    return ''.join([char for char in text if char not in string.punctuation])

# Reducing words to their root form using lemmatization
def lemmatization(text):
    return ' '.join([lm.lemmatize(word, pos='v') for word in text.split()])

# Main data processing function
def data_processing(text):
    text = str(text).lower()  # Convert text to lowercase
    text = replace_url(text)  # Replace URLs with 'URL'
    text = remove_html(text)  # Remove HTML tags
    text = replace_mentions(text)  # Replace mentions with 'user'
    text = replace_num(text)  # Replace numbers with 'NUMBER'
    text = replace_heart(text)  # Replace '<3' with 'HEART'
    text = remove_alphanumeric(text)  # Remove alphanumeric words
    text = remove_stopwords(text)  # Remove stopwords
    text = remove_punctuations(text)  # Remove punctuation
    text = lemmatization(text)  # Lemmatize words
    return text
