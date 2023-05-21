import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import re
import demoji
import string
from nltk.corpus import stopwords
from datetime import datetime, timedelta
from datetime import date
from textblob import TextBlob
import cleantext
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
import plotly.express as px


# Load the trained model
data = pd.read_csv('tweets.csv')

model = pickle.load(open('xgb_model.pkl', 'rb'))

st.title('RHONairobi Tweet Classification')

with st.expander('Analyze Text'):
    text = st.text_input('Text here:')
    if text:
        blob = TextBlob(text)
        st.write('Polarity:', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity:', round(blob.sentiment.subjectivity, 2))

pre = st.text_input('Clean Text:')
if pre:
    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                             stopwords=True, lowercase=True, numbers=True, punct=True, stemming=True))
with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload File')
    
    def score(a):
        blob1 = TextBlob(a)
        return blob1.sentiment.polarity
    
    def analyze(a):
        if a < 0:
            return 'Negative'
        elif a == 0:
            return 'Neutral'
        else:
            return 'Positive'
    
    if upl:
        df = pd.read_csv(upl)
        df['score'] = df['Text'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head())

        analysis_counts = df['analysis'].value_counts()

        fig = px.bar(analysis_counts, x=analysis_counts.index, y=analysis_counts.values, labels={'x': 'Sentiment Analysis', 'y': 'Count'},
             title='Sentiment Analysis Distribution')

# Display the chart using st.plotly_chart()
        st.plotly_chart(fig)
        
# Set up the Streamlit app
st.title('Real-time Tweet Analysis')

# Function to fetch and analyze tweets
def fetch_tweets(query, num_tweets):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append(tweet.content)
        if len(tweets) == num_tweets:
            break
    return tweets

# Get user input for query and number of tweets
query = st.text_input("#RHONairobi since:2023-02-01 until:2023-03-15")
num_tweets = st.number_input('5:', min_value=1, step=1)

# Fetch and analyze tweets
if query and num_tweets:
    tweets = fetch_tweets(query, num_tweets)
    df = pd.DataFrame({'Tweets': tweets})
    st.write(df)


        

       