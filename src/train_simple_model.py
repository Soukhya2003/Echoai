import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load data
print("Loading data...")
df = pd.read_excel("EchoAI_Chirper_dataset.xlsx")

# Preprocess text to create cleaned_text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

print("Preprocessing text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Label encoding
print("Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(df['agent_type'])

# TF-IDF
print("Creating TF-IDF...")
tfidf = TfidfVectorizer(max_features=200, ngram_range=(1,2))
X = tfidf.fit_transform(df['cleaned_text'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save models
joblib.dump(rf, "rf_simple.pkl")
joblib.dump(tfidf, "tfidf_simple.pkl")
joblib.dump(le, "label_encoder_simple.pkl")

print(f"✅ Models saved! Accuracy: {rf.score(X_test, y_test):.3f}")
print("Features used:", rf.n_features_in_)
