import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK resources
nltk.download('stopwords')

# Initialize the stemmer and stopwords set
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Define the function to load and train the model
def load_and_train_model():
    data = pd.read_csv("train.csv")
    data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive", 2: "Normal"})
    data = data[["tweet", "labels"]]

    def clean(text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = [word for word in text.split() if word not in stopwords_set]
        text = [stemmer.stem(word) for word in text]
        return " ".join(text)

    data["tweet"] = data["tweet"].apply(clean)
    X = np.array(data["tweet"])
    y = np.array(data["labels"])
    cv = CountVectorizer()
    X_vec = cv.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.33, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf, cv

# Load model and vectorizer
clf, cv = load_and_train_model()

# Streamlit app interface
st.title("Hate Speech Detection")

sample = st.text_area("Enter a tweet to classify:")

if st.button("Classify"):
    if sample:
        data_vec = cv.transform([sample]).toarray()
        prediction = clf.predict(data_vec)
        st.write(f"Prediction: {prediction[0]}")
    else:
        st.write("Please enter some text to classify.")
