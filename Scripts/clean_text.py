import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'@\w+', '', s)
    s = re.sub(r'[^a-z0-9\s#]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def clean_and_featurize(s):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # 1. lowercase & remove URLs/mentions
    s = str(s).lower()
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'@\w+', '', s)

    # 2. handle emojis
    s = emoji.demojize(s)

    # 3. remove non-alphanumeric (keep hashtags)
    s = re.sub(r'[^a-z0-9\s#]', ' ', s)

    # 4. tokenize & remove stopwords/short tokens
    tokens = [tok for tok in s.split() if tok not in stop_words and len(tok) > 2]

    # 5. lemmatize
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    return " ".join(tokens)