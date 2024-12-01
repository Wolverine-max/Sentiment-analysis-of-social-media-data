import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import numpy as np


st.title('Sentiment Analysis App')

st.write('This app is for sentiment analysis of social media data !')
st.sidebar.title('Sentiment analysis of Social Media Data')
with st.expander('Twitter Data'):
  st.write('**Raw data**')
  df1=pd.read_csv('https://raw.githubusercontent.com/Wolverine-max/Sentiment-analysis-of-social-media-data/refs/heads/master/Twitter_Data.csv')
  df1
with st.expander('Reddit Data'):
  st.write('**Raw data**')
  df2=pd.read_csv('https://raw.githubusercontent.com/Wolverine-max/Sentiment-analysis-of-social-media-data/refs/heads/master/Reddit_Data.csv')
  df2
st.sidebar.subheader('Data Analyser')

