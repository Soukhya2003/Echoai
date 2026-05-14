"""
Phase 3: Reorganize project structure
Run AFTER pipeline.py has completed.
py -3.11 reorganize.py
"""

import os, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))

dirs = ['data', 'models', 'figures', 'src', 'notebooks']
for d in dirs:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

# ── Move data files ────────────────────────────────────────────
DATA_FILES = [
    'EchoAI_Chirper_dataset.xlsx',
    'EchoAI_Chirper_dataset_BERT.xlsx',
    'EchoAI_Chirper_dataset_FINAL.xlsx',
    'sample_data.csv',
    'daily_trends.csv',
    'network_centrality.csv',
]
for f in DATA_FILES:
    src = os.path.join(ROOT, f)
    dst = os.path.join(ROOT, 'data', f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"Moved → data/{f}")

# ── Move model files ───────────────────────────────────────────
MODEL_FILES = [
    'rf_model.pkl', 'rf_simple.pkl', 'rf_sentiment.pkl', 'rf_polarization.pkl',
    'tfidf.pkl', 'tfidf_simple.pkl', 'tfidf_matrix_sample.pkl',
    'label_encoder.pkl', 'label_encoder_simple.pkl',
    'label_encoder_sentiment.pkl', 'scaler.pkl', 'hand_cols.pkl', 'topic_words.pkl',
    'lda_model', 'lda_model.expElogbeta.npy', 'lda_model.id2word',
    'lda_model.state', 'dictionary.gensim', 'dictionary.gensim.index',
    'network_graph.pkl',
]
for f in MODEL_FILES:
    src = os.path.join(ROOT, f)
    dst = os.path.join(ROOT, 'models', f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"Moved → models/{f}")

# ── Move source scripts ────────────────────────────────────────
SRC_FILES = [
    'train_bert.py', 'train_simple_model.py',
    'prepare_data.py', 'create_data.py', 'create_sample_data.py',
]
for f in SRC_FILES:
    src = os.path.join(ROOT, f)
    dst = os.path.join(ROOT, 'src', f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"Moved → src/{f}")

# ── Move notebook ──────────────────────────────────────────────
NB = 'Echo AI Project (3).ipynb'
src = os.path.join(ROOT, NB)
dst = os.path.join(ROOT, 'notebooks', NB)
if os.path.exists(src) and not os.path.exists(dst):
    shutil.move(src, dst)
    print(f"Moved → notebooks/{NB}")

# ── Update app.py paths ────────────────────────────────────────
app_path = os.path.join(ROOT, 'app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    '"EchoAI_Chirper_dataset_FINAL.xlsx"': '"data/EchoAI_Chirper_dataset_FINAL.xlsx"',
    '"EchoAI_Chirper_dataset_BERT.xlsx"':  '"data/EchoAI_Chirper_dataset_BERT.xlsx"',
    "'daily_trends.csv'":                  "'data/daily_trends.csv'",
    "'network_centrality.csv'":            "'data/network_centrality.csv'",
    "'lda_model'":                         "'models/lda_model'",
    "'dictionary.gensim'":                 "'models/dictionary.gensim'",
    "'topic_words.pkl'":                   "'models/topic_words.pkl'",
    "'network_graph.pkl'":                 "'models/network_graph.pkl'",
    "'network_interactive.html'":          "'network_interactive.html'",
}
for old, new in replacements.items():
    content = content.replace(old, new)

with open(app_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated app.py paths")

# ── Update api.py paths ────────────────────────────────────────
api_path = os.path.join(ROOT, 'api.py')
with open(api_path, 'r', encoding='utf-8') as f:
    content = f.read()

api_replacements = {
    '"rf_simple.pkl"':               '"models/rf_simple.pkl"',
    '"tfidf_simple.pkl"':            '"models/tfidf_simple.pkl"',
    '"label_encoder_simple.pkl"':    '"models/label_encoder_simple.pkl"',
    '"sample_data.csv"':             '"data/sample_data.csv"',
}
for old, new in api_replacements.items():
    content = content.replace(old, new)

with open(api_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated api.py paths")

print("\nReorganization complete.")
print("Run: py -3.11 -m streamlit run app.py")
print("Run: py -3.11 -m uvicorn api:app --reload")
