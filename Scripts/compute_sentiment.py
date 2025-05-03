from textblob import TextBlob
import pandas as pd

def analyze_sentiment(s):
    tb = TextBlob(s)
    return pd.Series([tb.polarity, tb.subjectivity])   