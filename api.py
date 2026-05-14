from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI(title="Echo AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load full-feature models (TF-IDF + VADER + text stats = 211 features) ──
rf     = joblib.load("models/rf_model.pkl")
tfidf  = joblib.load("models/tfidf.pkl")
le     = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
hand_cols = joblib.load("models/hand_cols.pkl")

# ── Load sample data for similarity search (use tfidf_simple for similarity) ──
tfidf_sim   = joblib.load("models/tfidf_simple.pkl")
df_sample   = pd.read_csv("data/sample_data.csv")
vader       = SentimentIntensityAnalyzer()
stop_words  = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join([t for t in tokens if t not in stop_words and len(t) > 2])

# Pre-build similarity matrix
print("Building similarity matrix...")
cleaned_samples = df_sample['cleaned_text'].astype(str) if 'cleaned_text' in df_sample.columns \
    else df_sample['text'].astype(str).apply(preprocess_text)
tfidf_matrix_sample = tfidf_sim.transform(cleaned_samples)
print(f"Similarity matrix shape: {tfidf_matrix_sample.shape}")

# Dataset means used as defaults for unknown engagement metrics
DEFAULTS = {
    'bert_score':       0.68,
    'likes':            250.6,
    'shares':           49.6,
    'engagement_score': 300.2,
}

CATEGORY_META = {
    "extremist":    {"emoji": "🔴", "color": "#ef4444", "desc": "Content promoting radical or violent ideologies"},
    "conservative": {"emoji": "🟠", "color": "#f97316", "desc": "Content reflecting traditional or right-leaning views"},
    "moderate":     {"emoji": "🟡", "color": "#eab308", "desc": "Content with balanced, centrist perspectives"},
    "neutral":      {"emoji": "⚪", "color": "#94a3b8", "desc": "Objective content without political leaning"},
    "progressive":  {"emoji": "🔵", "color": "#3b82f6", "desc": "Content reflecting reform-oriented or left-leaning views"},
}

def extract_features(text: str, cleaned: str):
    """Build the full 211-feature vector: 200 TF-IDF + 11 handcrafted."""
    # TF-IDF features
    X_tfidf = tfidf.transform([cleaned])

    # Handcrafted features
    vs      = vader.polarity_scores(text)
    tokens  = word_tokenize(cleaned)
    words   = cleaned.split()

    feat_map = {
        'sentiment_compound': vs['compound'],
        'bert_score':         DEFAULTS['bert_score'],
        'likes':              DEFAULTS['likes'],
        'shares':             DEFAULTS['shares'],
        'engagement_score':   DEFAULTS['engagement_score'],
        'text_length':        len(text),
        'word_count':         len(words),
        'token_count':        len(tokens),
        'hashtag_count':      text.count('#'),
        'mention_count':      text.count('@'),
        'avg_word_length':    np.mean([len(w) for w in words]) if words else 0,
    }

    raw_hand = np.array([[feat_map[c] for c in hand_cols]], dtype=float)
    X_hand   = csr_matrix(scaler.transform(raw_hand))
    return hstack([X_tfidf, X_hand])


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict(input: TextInput):
    cleaned = preprocess_text(input.text)

    # Build full feature vector
    features = extract_features(input.text, cleaned)

    pred       = rf.predict(features)[0]
    pred_label = le.inverse_transform([pred])[0].lower()
    probs      = rf.predict_proba(features)[0].tolist()

    prob_dict = {cls.lower(): round(float(p) * 100, 1)
                 for cls, p in zip(le.classes_, probs)}

    # Similarity via the simple TF-IDF vectoriser
    sim_vec    = tfidf_sim.transform([cleaned])
    sim_scores = cosine_similarity(sim_vec, tfidf_matrix_sample)[0]
    top_idx    = np.argsort(sim_scores)[-5:][::-1]
    similar_posts = [
        {
            "text":       df_sample.iloc[i]['text'][:250],
            "agent_type": df_sample.iloc[i]['agent_type'].lower(),
            "similarity": round(float(sim_scores[i]) * 100, 1),
        }
        for i in top_idx if sim_scores[i] > 0.05
    ]

    meta = CATEGORY_META.get(pred_label, {})
    vs   = vader.polarity_scores(input.text)

    return {
        "predicted_agent": pred_label,
        "emoji":           meta.get("emoji", "?"),
        "color":           meta.get("color", "#fff"),
        "description":     meta.get("desc", ""),
        "probabilities":   prob_dict,
        "similar_posts":   similar_posts,
        "vader_compound":  round(vs['compound'], 3),
    }


@app.get("/health")
def health():
    return {"status": "ok", "message": "Echo AI API is running"}


@app.get("/")
def root():
    return {"message": "Echo AI API - POST to /predict with {\"text\": \"...\"}"}
