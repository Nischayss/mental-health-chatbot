from flask import Flask, render_template, request
import os
import json
import joblib
import pickle
import numpy as np
import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(_name_)
app.static_folder = 'static'

# File paths
csv_path = "D:/chatbot/20200325_counsel_chat.csv"
question_embeddings_path = "D:/chatbot/model/toc_question_embeddings.dump"
answer_embeddings_path = "D:/chatbot/model/toc_answer_embeddings.dump"
answer_dict_path = "D:/chatbot/model/toc_topic_answers.pkl"

# Load data
df = pd.read_csv(csv_path, encoding="utf-8")
question_embeddings = joblib.load(question_embeddings_path)
answer_embeddings = joblib.load(answer_embeddings_path)

with open(answer_dict_path, "rb") as f:
    topic_answer_map = pickle.load(f)

# NLP tools
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Clean user text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# Emotion-trigger keywords (fallback when low similarity)
emotion_keywords = {
    'cry': "It's okay to cry. You're not alone. Want to talk about what's making you feel this way?",
    'unsafe': "I'm really sorry you're feeling unsafe. Your well-being matters. Can you reach out to someone you trust?",
    'failure': "Feeling like a failure is incredibly tough. Just know you're doing your best and you're not alone in this.",
    'anxiety': "Anxiety can feel overwhelming. I'm here for you. Would talking about it help?",
    'overwhelmed': "Take a deep breath. It's okay to feel overwhelmed. Want to talk through what's stressing you?",
    'tired': "If you're feeling exhausted, maybe we can figure out what's weighing on you. I'm listening.",
    'depressed': "I'm sorry you're feeling this way. You're not alone. Would you like to talk more about it?"
}

# Similarity + fallback logic
def retrieve_answer(user_query):
    cleaned_query = clean_text(user_query)
    query_vector = embedding_model.encode([cleaned_query])

    similarities = cosine_similarity(query_vector, question_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= 0.60:
        return df.iloc[best_idx]['answerText']

    # Fallback: match emotional keywords
    for keyword, response in emotion_keywords.items():
        if keyword in cleaned_query:
            return response

    return "I'm here for you. Could you tell me a bit more so I can understand better?"

# Greeting/exit responses
greetings = ['hi', 'hey', 'hello', 'good morning', 'good evening']
goodbyes = ['bye', 'thank you', 'thanks', 'goodbye']

# Load model only once
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Web routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    cleaned = clean_text(user_text)

    if any(greet in cleaned for greet in greetings):
        return "Hello! How may I assist you today?"
    elif any(bye in cleaned for bye in goodbyes):
        return "Take care. I'm here if you need me anytime."

    return retrieve_answer(user_text)

if _name_ == "_main_":
    app.run(debug=True)