#!/usr/bin/env python3
"""
EchoAI Analysis Pipeline - Steps 6-10
LDA Topics | Clustering | Classification | Network Analysis | Time Series
Run: py -3.11 pipeline.py
"""

import os, sys, warnings, pickle
warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
df = pd.read_excel('data/EchoAI_Chirper_dataset_BERT.xlsx')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")
print(f"Agent types: {df['agent_type'].value_counts().to_dict()}")

# ============================================================
print("\n" + "=" * 60)
print("STEP 6: LDA TOPIC MODELLING")
print("=" * 60)

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
import re
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from wordcloud import WordCloud

STOP = set(stopwords.words('english'))
EXTRA_STOP = {
    'said', 'also', 'would', 'could', 'should', 'must', 'like', 'just',
    'one', 'even', 'make', 'get', 'use', 'say', 'think', 'know', 'want',
    'time', 'way', 'people', 'good', 'new', 'year', 'day', 'thing', 'well',
    'still', 'need', 'many', 'first', 'take', 'see', 'much', 'come'
}
STOP = STOP | EXTRA_STOP

def tokenize_lda(text):
    text = re.sub(r'[^a-z\s]', '', str(text).lower())
    return [t for t in text.split() if t not in STOP and len(t) > 2]

print("Tokenizing text for LDA...")
df['lda_tokens'] = df['cleaned_text'].apply(tokenize_lda)

dictionary = Dictionary(df['lda_tokens'])
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(t) for t in df['lda_tokens']]
print(f"Dictionary: {len(dictionary)} tokens | Corpus: {len(corpus)} docs")

NUM_TOPICS = 5
print(f"Training LDA ({NUM_TOPICS} topics, 15 passes)...")
lda = LdaModel(
    corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS,
    passes=15, random_state=42, alpha='auto', per_word_topics=True
)

try:
    # u_mass is single-threaded and works on Windows without multiprocessing issues
    coherence = CoherenceModel(
        model=lda, texts=df['lda_tokens'], dictionary=dictionary, coherence='u_mass'
    ).get_coherence()
    print(f"Coherence Score (u_mass): {coherence:.4f}")
except Exception as ce:
    print(f"Coherence computation skipped: {ce}")

for i in range(NUM_TOPICS):
    words = [w for w, _ in lda.show_topic(i, 8)]
    print(f"  Topic {i}: {', '.join(words)}")

def get_dominant(bow):
    topics = lda.get_document_topics(bow)
    if not topics:
        return 0, 0.0
    best = max(topics, key=lambda x: x[1])
    return int(best[0]), float(best[1])

print("Assigning dominant topics...")
results = [get_dominant(bow) for bow in tqdm(corpus)]
df['dominant_topic'] = [r[0] for r in results]
df['topic_confidence'] = [r[1] for r in results]

# Figures: topic distribution + agent heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tc = df['dominant_topic'].value_counts().sort_index()
axes[0].bar(tc.index, tc.values, color=sns.color_palette('tab10', NUM_TOPICS))
axes[0].set_xlabel('Topic'); axes[0].set_ylabel('Posts'); axes[0].set_title('Posts per Topic')

ta = pd.crosstab(df['dominant_topic'], df['agent_type'])
sns.heatmap(ta, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Topic x Agent Type Heatmap')
plt.tight_layout()
plt.savefig('figures/lda_topics.png', dpi=150, bbox_inches='tight')
plt.close()

# Word clouds
fig, axes = plt.subplots(1, NUM_TOPICS, figsize=(20, 4))
for i in range(NUM_TOPICS):
    wc = WordCloud(width=400, height=300, background_color='white')\
        .generate_from_frequencies(dict(lda.show_topic(i, 50)))
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(f'Topic {i}', fontsize=10)
plt.suptitle('LDA Topic Word Clouds', fontsize=14)
plt.tight_layout()
plt.savefig('figures/lda_wordclouds.png', dpi=150, bbox_inches='tight')
plt.close()

lda.save('models/lda_model')
dictionary.save('models/dictionary.gensim')

# Save topic top words for dashboard
topic_words = {i: [w for w, _ in lda.show_topic(i, 20)] for i in range(NUM_TOPICS)}
joblib.dump(topic_words, 'models/topic_words.pkl')

df.drop(columns=['lda_tokens'], inplace=True, errors='ignore')
df.to_excel('data/EchoAI_Chirper_dataset_FINAL.xlsx', index=False)
print("STEP 6 DONE: LDA saved, dominant_topic/topic_confidence added")

# ============================================================
print("\n" + "=" * 60)
print("STEP 7: CLUSTERING")
print("=" * 60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse import hstack, csr_matrix

print("Building feature matrix...")
tfidf_clust = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
X_tfidf = tfidf_clust.fit_transform(df['cleaned_text'].fillna(''))

hand_cols = [c for c in ['sentiment_compound', 'bert_score', 'likes', 'shares',
                          'engagement_score', 'text_length', 'word_count',
                          'token_count', 'hashtag_count', 'mention_count']
             if c in df.columns]
scaler_c = StandardScaler()
X_hand = csr_matrix(scaler_c.fit_transform(df[hand_cols].fillna(0)))
X_all = hstack([X_tfidf, X_hand])
print(f"Feature matrix: {X_all.shape}")

print("Elbow + Silhouette (K=2 to 10)...")
inertias, sil_scores = [], []
for k in tqdm(range(2, 11)):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_all)
    inertias.append(km.inertia_)
    s = silhouette_score(X_all, km.labels_, sample_size=min(1000, len(df)), random_state=42)
    sil_scores.append(s)
    print(f"  K={k}: sil={s:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(2, 11), inertias, 'bo-'); ax1.set_title('Elbow'); ax1.set_xlabel('K')
ax2.plot(range(2, 11), sil_scores, 'ro-'); ax2.set_title('Silhouette'); ax2.set_xlabel('K')
plt.tight_layout()
plt.savefig('figures/clustering_elbow.png', dpi=150, bbox_inches='tight')
plt.close()

best_k = range(2, 11)[np.argmax(sil_scores)]
print(f"Best K by silhouette: {best_k} (score={max(sil_scores):.4f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_final.fit(X_all)
df['kmeans_cluster'] = km_final.labels_
print(f"Final silhouette: {silhouette_score(X_all, km_final.labels_, sample_size=min(1000,len(df)), random_state=42):.4f}")

# Hierarchical clustering on 200-post sample
np.random.seed(42)
samp_idx = np.random.choice(len(df), min(200, len(df)), replace=False)
X_samp = X_all[samp_idx].toarray()
Z = linkage(X_samp, method='ward')

fig, ax = plt.subplots(figsize=(16, 6))
dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7 * max(Z[:, 2]))
ax.set_title('Hierarchical Clustering Dendrogram (200-post sample)')
plt.tight_layout()
plt.savefig('figures/dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()

hier_labels = fcluster(Z, t=best_k, criterion='maxclust')
df['hier_cluster'] = -1
for i, idx in enumerate(samp_idx):
    df.at[idx, 'hier_cluster'] = int(hier_labels[i] - 1)

# PCA for visualization
print("PCA 2D projection...")
pca = PCA(n_components=2, random_state=42)
X_dense = X_all.toarray()
X_pca = pca.fit_transform(X_dense)
df['pca_x'] = X_pca[:, 0]
df['pca_y'] = X_pca[:, 1]
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

AGENT_COLORS = {
    'extremist': '#ef4444', 'conservative': '#f97316',
    'moderate': '#eab308', 'neutral': '#94a3b8', 'progressive': '#3b82f6'
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
cmap = plt.cm.tab10(np.linspace(0, 1, best_k))
for c in range(best_k):
    m = df['kmeans_cluster'] == c
    ax1.scatter(X_pca[m, 0], X_pca[m, 1], s=5, alpha=0.4, label=f'C{c}', color=cmap[c])
ax1.set_title(f'PCA – K-Means (K={best_k})'); ax1.legend(markerscale=3)

for at, col in AGENT_COLORS.items():
    m = df['agent_type'] == at
    ax2.scatter(X_pca[m, 0], X_pca[m, 1], s=5, alpha=0.4, color=col, label=at)
ax2.set_title('PCA – Agent Type'); ax2.legend(markerscale=3)
plt.tight_layout()
plt.savefig('figures/clustering_pca.png', dpi=150, bbox_inches='tight')
plt.close()

df.to_excel('data/EchoAI_Chirper_dataset_FINAL.xlsx', index=False)
print("STEP 7 DONE: kmeans_cluster, hier_cluster, pca_x, pca_y added")

# ============================================================
print("\n" + "=" * 60)
print("STEP 8: CLASSIFICATION")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, ConfusionMatrixDisplay)

# Build features for classification
tfidf_clf = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
X_clf_tfidf = tfidf_clf.fit_transform(df['cleaned_text'].fillna(''))

hand_clf = [c for c in ['sentiment_compound', 'bert_score', 'likes', 'shares',
                         'engagement_score', 'text_length', 'word_count',
                         'token_count', 'hashtag_count', 'mention_count', 'avg_word_length']
            if c in df.columns]
scaler_clf = StandardScaler()
X_clf_hand = csr_matrix(scaler_clf.fit_transform(df[hand_clf].fillna(0)))
X_clf = hstack([X_clf_tfidf, X_clf_hand])
print(f"Classification features: {X_clf.shape}  ({len(hand_clf)} handcrafted)")

PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

# --- Task 1: Sentiment Classification ---
print("\n--- Task 1: Sentiment Classification ---")
le_sent = LabelEncoder()
y_sent = le_sent.fit_transform(df['sentiment_label'])
print(f"Classes: {le_sent.classes_}  |  Dist: {dict(zip(le_sent.classes_, np.bincount(y_sent)))}")

Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
    X_clf, y_sent, test_size=0.2, random_state=42, stratify=y_sent)

lr_s = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_s.fit(Xtr_s, ytr_s)
print(f"LR baseline accuracy: {lr_s.score(Xte_s, yte_s):.3f}")

print("Grid-searching RF (sentiment)...")
rf_gs_s = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    PARAM_GRID, cv=3, scoring='f1_macro', n_jobs=-1, verbose=0)
rf_gs_s.fit(Xtr_s, ytr_s)
y_pred_s = rf_gs_s.predict(Xte_s)
print(f"Best params: {rf_gs_s.best_params_}")
print(classification_report(yte_s, y_pred_s, target_names=le_sent.classes_))

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix(yte_s, y_pred_s), display_labels=le_sent.classes_).plot(ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix – Sentiment Classification')
plt.tight_layout(); plt.savefig('figures/cm_sentiment.png', dpi=150, bbox_inches='tight'); plt.close()

feat_names = tfidf_clf.get_feature_names_out().tolist() + hand_clf
imps = rf_gs_s.best_estimator_.feature_importances_
top20 = np.argsort(imps)[-20:][::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([feat_names[i] for i in top20], imps[top20])
ax.set_title('Top 20 Features – Sentiment')
plt.tight_layout(); plt.savefig('figures/fi_sentiment.png', dpi=150, bbox_inches='tight'); plt.close()

joblib.dump(rf_gs_s.best_estimator_, 'models/rf_sentiment.pkl')
joblib.dump(le_sent, 'models/label_encoder_sentiment.pkl')
print("Sentiment model saved: rf_sentiment.pkl")

# --- Task 2: Polarization Detection ---
print("\n--- Task 2: Polarization Detection ---")
y_pol = df['is_polarized'].values
print(f"Balance: {np.bincount(y_pol)}  |  Extremist: {(df['agent_type']=='extremist').sum()}")

Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(
    X_clf, y_pol, test_size=0.2, random_state=42, stratify=y_pol)

lr_p = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_p.fit(Xtr_p, ytr_p)
print(f"LR baseline accuracy: {lr_p.score(Xte_p, yte_p):.3f}")

print("Grid-searching RF (polarization)...")
rf_gs_p = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    PARAM_GRID, cv=3, scoring='f1', n_jobs=-1, verbose=0)
rf_gs_p.fit(Xtr_p, ytr_p)
y_pred_p = rf_gs_p.predict(Xte_p)
print(f"Best params: {rf_gs_p.best_params_}")
print(classification_report(yte_p, y_pred_p, target_names=['Not Polarized', 'Polarized']))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(yte_p, y_pred_p), display_labels=['Not Pol.', 'Polarized']).plot(ax=ax, cmap='Reds')
ax.set_title('Confusion Matrix – Polarization Detection')
plt.tight_layout(); plt.savefig('figures/cm_polarization.png', dpi=150, bbox_inches='tight'); plt.close()

y_prob_p = rf_gs_p.predict_proba(Xte_p)[:, 1]
fpr, tpr, _ = roc_curve(yte_p, y_prob_p)
auc = roc_auc_score(yte_p, y_prob_p)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}'); ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC – Polarization'); ax.legend()
plt.tight_layout(); plt.savefig('figures/roc_polarization.png', dpi=150, bbox_inches='tight'); plt.close()

imps_p = rf_gs_p.best_estimator_.feature_importances_
top20_p = np.argsort(imps_p)[-20:][::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([feat_names[i] for i in top20_p], imps_p[top20_p])
ax.set_title('Top 20 Features – Polarization')
plt.tight_layout(); plt.savefig('figures/fi_polarization.png', dpi=150, bbox_inches='tight'); plt.close()

joblib.dump(rf_gs_p.best_estimator_, 'models/rf_polarization.pkl')
print("Polarization model saved: rf_polarization.pkl")

# --- Task 3: Agent Type (for API) ---
print("\n--- Task 3: Agent Type Classification (for API) ---")
le_agent = LabelEncoder()
y_agent = le_agent.fit_transform(df['agent_type'])

Xtr_a, Xte_a, ytr_a, yte_a = train_test_split(
    X_clf, y_agent, test_size=0.2, random_state=42, stratify=y_agent)

rf_agent = RandomForestClassifier(
    n_estimators=200, max_depth=20, random_state=42,
    class_weight='balanced', n_jobs=-1)
rf_agent.fit(Xtr_a, ytr_a)
print(f"Agent type accuracy: {rf_agent.score(Xte_a, yte_a):.3f}")
print(classification_report(yte_a, rf_agent.predict(Xte_a), target_names=le_agent.classes_))

joblib.dump(rf_agent, 'models/rf_model.pkl')
joblib.dump(le_agent, 'models/label_encoder.pkl')
joblib.dump(tfidf_clf, 'models/tfidf.pkl')
joblib.dump(scaler_clf, 'models/scaler.pkl')
joblib.dump(hand_clf, 'models/hand_cols.pkl')
print("API models saved: rf_model.pkl, label_encoder.pkl, tfidf.pkl")

# Rebuild sample data + matrix for API
print("Rebuilding sample_data.csv and tfidf_matrix_sample.pkl...")
df_samp = df[['text', 'cleaned_text', 'agent_type']].dropna().head(2000)
df_samp.to_csv('data/sample_data.csv', index=False)
tfidf_mat = tfidf_clf.transform(df_samp['cleaned_text'].fillna(''))
joblib.dump(tfidf_mat, 'models/tfidf_matrix_sample.pkl')
print(f"sample_data: {len(df_samp)} rows | tfidf_matrix: {tfidf_mat.shape}")

# Combined RF model report
fig, ax = plt.subplots(figsize=(9, 7))
cm_a = confusion_matrix(yte_a, rf_agent.predict(Xte_a))
ConfusionMatrixDisplay(cm_a, display_labels=le_agent.classes_).plot(ax=ax, cmap='Greens')
ax.set_title('Confusion Matrix – Agent Type Classification')
plt.tight_layout(); plt.savefig('figures/cm_agent_type.png', dpi=150, bbox_inches='tight'); plt.close()

print("STEP 8 DONE")

# ============================================================
print("\n" + "=" * 60)
print("STEP 9: NETWORK ANALYSIS")
print("=" * 60)

import networkx as nx
import community as community_louvain

post_to_agent = dict(zip(df['post_id'], df['agent_id']))
df_rep = df[df['reply_to'].notna()].copy()
print(f"Reply interactions: {len(df_rep)}")

G = nx.DiGraph()
agent_meta = df.drop_duplicates('agent_id')[['agent_id', 'agent_type', 'agent_name']].set_index('agent_id')
for aid, row in agent_meta.iterrows():
    G.add_node(aid, agent_type=row['agent_type'], agent_name=row['agent_name'])

edges = 0
for _, row in df_rep.iterrows():
    src = row['agent_id']
    dst_post = row['reply_to']
    if dst_post in post_to_agent:
        dst = post_to_agent[dst_post]
        if src != dst:
            if G.has_edge(src, dst):
                G[src][dst]['weight'] += 1
            else:
                G.add_edge(src, dst, weight=1)
            edges += 1

print(f"Graph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges | {edges} interactions")

# Centrality metrics
deg_c = nx.degree_centrality(G)
bet_c = nx.betweenness_centrality(G, normalized=True, weight='weight')
try:
    pr = nx.pagerank(G, weight='weight', max_iter=300)
except Exception:
    pr = {n: 1 / max(len(G), 1) for n in G.nodes()}
in_d = dict(G.in_degree(weight='weight'))
out_d = dict(G.out_degree(weight='weight'))

cent_df = pd.DataFrame({
    'agent_id':    list(G.nodes()),
    'agent_name':  [G.nodes[n].get('agent_name', '') for n in G.nodes()],
    'agent_type':  [G.nodes[n].get('agent_type', '') for n in G.nodes()],
    'degree':      [deg_c.get(n, 0) for n in G.nodes()],
    'betweenness': [bet_c.get(n, 0) for n in G.nodes()],
    'pagerank':    [pr.get(n, 0) for n in G.nodes()],
    'in_degree':   [in_d.get(n, 0) for n in G.nodes()],
    'out_degree':  [out_d.get(n, 0) for n in G.nodes()],
}).sort_values('betweenness', ascending=False)

print("\nTop 10 influential agents (betweenness):")
print(cent_df.head(10)[['agent_name', 'agent_type', 'betweenness', 'pagerank', 'in_degree']].to_string(index=False))
cent_df.to_csv('data/network_centrality.csv', index=False)

# Community detection
G_un = G.to_undirected()
if len(G_un.edges()) > 0:
    comms = community_louvain.best_partition(G_un, random_state=42)
    n_comms = len(set(comms.values()))
    print(f"Louvain communities: {n_comms}")
    nx.set_node_attributes(G, comms, 'community')
else:
    comms = {n: 0 for n in G.nodes()}
    print("No edges for community detection")

# Network visualization (static)
fig, ax = plt.subplots(figsize=(14, 10))
nc = [AGENT_COLORS.get(G.nodes[n].get('agent_type', ''), '#666') for n in G.nodes()]
ns = [max(30, min(600, deg_c.get(n, 0) * 3000)) for n in G.nodes()]
pos = nx.spring_layout(G, k=0.5, seed=42)
nx.draw_networkx(G, pos, ax=ax, node_color=nc, node_size=ns,
                 edge_color='#ccc', alpha=0.7, arrows=True,
                 arrowsize=8, width=0.5, with_labels=False)
for at, col in AGENT_COLORS.items():
    ax.scatter([], [], c=col, s=100, label=at.title())
ax.legend(title='Agent Type', loc='upper left')
ax.set_title('EchoAI Agent Interaction Network\n(node size ∝ degree centrality)', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig('figures/network_graph.png', dpi=150, bbox_inches='tight')
plt.close()

# Centrality boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
agent_order = ['extremist', 'conservative', 'moderate', 'neutral', 'progressive']
short = ['Ext', 'Con', 'Mod', 'Neu', 'Pro']
for i, (metric, title) in enumerate([('degree', 'Degree'), ('betweenness', 'Betweenness'), ('pagerank', 'PageRank')]):
    data = [cent_df[cent_df['agent_type'] == at][metric].values for at in agent_order]
    axes[i].boxplot(data, labels=short, patch_artist=True)
    axes[i].set_title(title)
plt.suptitle('Centrality by Agent Type', fontsize=13)
plt.tight_layout()
plt.savefig('figures/centrality_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

# Save graph for dashboard
with open('models/network_graph.pkl', 'wb') as f:
    pickle.dump({'G': G, 'pos': pos, 'cent_df': cent_df}, f)

# Pyvis interactive HTML
try:
    from pyvis.network import Network
    net = Network(height='600px', width='100%', directed=True, bgcolor='#fff', font_color='#333')
    net.barnes_hut(spring_length=200, central_gravity=0.3)
    color_map = {0: '#ef4444', 1: '#f97316', 2: '#eab308', 3: '#94a3b8', 4: '#3b82f6',
                 'extremist': '#ef4444', 'conservative': '#f97316', 'moderate': '#eab308',
                 'neutral': '#94a3b8', 'progressive': '#3b82f6'}
    for node in G.nodes(data=True):
        nid, attrs = node
        at = attrs.get('agent_type', 'neutral')
        col = AGENT_COLORS.get(at, '#888')
        sz = max(10, min(50, deg_c.get(nid, 0) * 500))
        net.add_node(nid, label=attrs.get('agent_name', nid),
                     color=col, size=sz, title=f"{attrs.get('agent_name','')}\n{at}")
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d.get('weight', 1))
    net.save_graph('network_interactive.html')
    print("Pyvis HTML saved: network_interactive.html")
except Exception as e:
    print(f"Pyvis skipped: {e}")

print("STEP 9 DONE")

# ============================================================
print("\n" + "=" * 60)
print("STEP 10: TIME SERIES POLARIZATION TRACKING")
print("=" * 60)

import pymannkendall as mk
import scipy.stats as stats

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
print(f"Valid timestamps: {df['timestamp'].notna().sum()}/{len(df)}")
print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

df['date'] = df['timestamp'].dt.date

daily = df.groupby('date').agg(
    total_posts=('post_id', 'count'),
    mean_vader=('sentiment_compound', 'mean'),
    mean_bert=('bert_score', 'mean'),
    extremist_posts=('agent_type', lambda x: (x == 'extremist').sum()),
    mean_engagement=('engagement_score', 'mean'),
).reset_index()

daily['date'] = pd.to_datetime(daily['date'])
daily = daily.sort_values('date').reset_index(drop=True)
daily['polarization_ratio'] = daily['extremist_posts'] / daily['total_posts']
daily['vader_ma3'] = daily['mean_vader'].rolling(3, min_periods=1).mean()
daily['bert_ma3'] = daily['mean_bert'].rolling(3, min_periods=1).mean()
daily['polar_ma3'] = daily['polarization_ratio'].rolling(3, min_periods=1).mean()

print(f"Days with data: {len(daily)}")
print(f"Polarization ratio: {daily['polarization_ratio'].mean():.3f} mean, {daily['polarization_ratio'].std():.3f} std")

# Mann-Kendall tests
print("\nMann-Kendall Trend Tests:")
for col, label in [('polarization_ratio', 'Polarization'), ('mean_vader', 'VADER Sentiment'), ('mean_bert', 'BERT Score')]:
    s = daily[col].dropna()
    if len(s) >= 4:
        r = mk.original_test(s)
        print(f"  {label:18s}: trend={r.trend:14s}  p={r.p:.4f}  tau={r.Tau:.3f}")

# Cohen's d (first vs second half)
def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0

n = len(daily)
d = cohens_d(daily['polarization_ratio'][n//2:].values, daily['polarization_ratio'][:n//2].values)
print(f"\nCohen's d (polarization 2nd half vs 1st half): {d:.4f}")

# Figure 1: Dual-axis
fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()
ax1.plot(daily['date'], daily['vader_ma3'], 'b-', lw=2, label='VADER (3-day MA)')
ax1.plot(daily['date'], daily['bert_ma3'], 'g--', lw=2, label='BERT (3-day MA)')
ax2.plot(daily['date'], daily['polar_ma3'], 'r-', lw=2, label='Polarization (3-day MA)')
ax1.set_ylabel('Sentiment Score', color='blue')
ax2.set_ylabel('Polarization Ratio', color='red')
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc='upper left')
plt.title('EchoAI: Sentiment & Polarization Trends')
plt.tight_layout()
plt.savefig('figures/timeseries_trends.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Stacked area
daily_at = df.groupby(['date', 'agent_type']).size().unstack(fill_value=0)
daily_at.index = pd.to_datetime(daily_at.index)
daily_at = daily_at.sort_index()
at_order = [c for c in ['extremist', 'conservative', 'moderate', 'neutral', 'progressive'] if c in daily_at.columns]
fig, ax = plt.subplots(figsize=(14, 5))
ax.stackplot(daily_at.index, [daily_at[c] for c in at_order],
             labels=[c.title() for c in at_order],
             colors=[AGENT_COLORS[c] for c in at_order], alpha=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Posts'); ax.set_title('Daily Posting by Agent Type')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('figures/timeseries_stacked.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Heatmap
hmap = daily_at.T
if len(hmap.columns) > 60:
    hmap = hmap.T.resample('W').sum().T
fig, ax = plt.subplots(figsize=(16, 4))
sns.heatmap(hmap, cmap='YlOrRd', ax=ax, linewidths=0, cbar_kws={'label': 'Posts'})
ax.set_title('Activity Heatmap (Agent Type × Time)')
plt.tight_layout()
plt.savefig('figures/timeseries_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

daily.to_csv('data/daily_trends.csv', index=False)
print("STEP 10 DONE")

# ============================================================
print("\n" + "=" * 60)
print("SAVING FINAL DATASET")
print("=" * 60)

df.drop(columns=['date'], inplace=True, errors='ignore')
df.to_excel('data/EchoAI_Chirper_dataset_FINAL.xlsx', index=False)
print(f"Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"New columns: {[c for c in df.columns if c not in pd.read_excel('data/EchoAI_Chirper_dataset_BERT.xlsx').columns]}")

figs = os.listdir('figures')
print(f"Figures saved: {len(figs)} files in figures/")
for f in sorted(figs):
    print(f"  - {f}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print("Dataset:          EchoAI_Chirper_dataset_FINAL.xlsx")
print("LDA Model:        lda_model + dictionary.gensim")
print("Classification:   rf_model.pkl, rf_sentiment.pkl, rf_polarization.pkl")
print("Network:          network_graph.pkl, network_centrality.csv")
print("Time Series:      daily_trends.csv")
print("API assets:       sample_data.csv, tfidf_matrix_sample.pkl")
print("=" * 60)
