import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    tokens = text.lower().split()
    tokens = [x for x in tokens if x not in stop_words and x not in string.punctuation]
    return tokens

#This is the primary preprocesser for the project
def preprocessv2(text):
    text = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text)
    tokens = [x for x in tokens if x not in stop_words]
    tokens = [lemmatizer.lemmatize(x) for x in tokens]
    return tokens
