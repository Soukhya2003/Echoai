"""
EchoAI Streamlit Dashboard
Multi-page analysis of AI agent polarization on Chirper.AI
Run: py -3.11 -m streamlit run app.py
"""

import os, warnings, pickle
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# NLTK setup (auto-download for hosted environments)
import nltk
for pkg in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="EchoAI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ───────────────────────────────────────────────
DATA_FILE = "data/EchoAI_Chirper_dataset_FINAL.xlsx"
FALLBACK   = "data/EchoAI_Chirper_dataset_BERT.xlsx"

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    path = DATA_FILE if os.path.exists(DATA_FILE) else FALLBACK
    df = pd.read_excel(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'].dt.date
    return df

@st.cache_data(show_spinner="Loading trend data…")
def load_daily():
    if os.path.exists('data/daily_trends.csv'):
        d = pd.read_csv('data/daily_trends.csv')
        d['date'] = pd.to_datetime(d['date'])
        return d
    return None

@st.cache_data(show_spinner="Loading network data…")
def load_network():
    if os.path.exists('data/network_centrality.csv'):
        return pd.read_csv('data/network_centrality.csv')
    return None

@st.cache_resource(show_spinner="Loading classifier model…")
def load_classifier():
    try:
        rf        = joblib.load("models/rf_model.pkl")
        tfidf     = joblib.load("models/tfidf.pkl")
        le        = joblib.load("models/label_encoder.pkl")
        scaler_m  = joblib.load("models/scaler.pkl")
        hand_cols = joblib.load("models/hand_cols.pkl")
        tfidf_sim = joblib.load("models/tfidf_simple.pkl")
        vader     = SentimentIntensityAnalyzer()
        stop_w    = set(stopwords.words('english'))

        if os.path.exists("data/sample_data.csv"):
            df_samp = pd.read_csv("data/sample_data.csv")
            cleaned_samples = df_samp['cleaned_text'].astype(str) if 'cleaned_text' in df_samp.columns \
                else df_samp['text'].astype(str)
            sim_matrix = tfidf_sim.transform(cleaned_samples)
        else:
            df_samp = None
            sim_matrix = None
        return rf, tfidf, le, scaler_m, hand_cols, tfidf_sim, vader, stop_w, df_samp, sim_matrix
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")
        return None, None, None, None, None, None, None, None, None, None

@st.cache_resource(show_spinner="Loading models…")
def load_lda():
    try:
        from gensim.models import LdaModel
        from gensim.corpora import Dictionary
        lda  = LdaModel.load('models/lda_model')
        d    = Dictionary.load('models/dictionary.gensim')
        tw   = joblib.load('models/topic_words.pkl') if os.path.exists('models/topic_words.pkl') else {}
        return lda, d, tw
    except Exception:
        return None, None, {}

df_full   = load_data()
daily_df  = load_daily()
cent_df   = load_network()

# ── Color maps ─────────────────────────────────────────────────
AGENT_COLORS = {
    'extremist': '#ef4444', 'conservative': '#f97316',
    'moderate': '#eab308',  'neutral': '#94a3b8',
    'progressive': '#3b82f6',
}
SENT_COLORS = {'positive': '#22c55e', 'negative': '#ef4444', 'neutral': '#94a3b8'}

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/robot-2.png", width=80)
st.sidebar.title("EchoAI")
st.sidebar.caption("AI Polarization Research Dashboard")
st.sidebar.divider()

PAGE = st.sidebar.radio(
    "Navigate",
    ["Overview", "Classifier", "Sentiment", "Topics", "Clustering",
     "Network", "Polarization", "Data Explorer"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.subheader("Filters")

all_agents = sorted(df_full['agent_type'].unique())
sel_agents = st.sidebar.multiselect("Agent Type", all_agents, default=all_agents)

all_sent = sorted(df_full['sentiment_label'].unique()) if 'sentiment_label' in df_full.columns else []
sel_sent = st.sidebar.multiselect("VADER Sentiment", all_sent, default=all_sent)

if df_full['timestamp'].notna().any():
    ts_min = df_full['timestamp'].min().date()
    ts_max = df_full['timestamp'].max().date()
    date_range = st.sidebar.slider(
        "Date Range",
        min_value=ts_min, max_value=ts_max,
        value=(ts_min, ts_max),
    )
else:
    date_range = None

search_text = st.sidebar.text_input("Search posts", placeholder="keyword…")

# Apply filters
df = df_full.copy()
if sel_agents:
    df = df[df['agent_type'].isin(sel_agents)]
if sel_sent and 'sentiment_label' in df.columns:
    df = df[df['sentiment_label'].isin(sel_sent)]
if date_range and df['timestamp'].notna().any():
    df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
if search_text.strip():
    mask = df['text'].str.contains(search_text.strip(), case=False, na=False)
    df = df[mask]

st.sidebar.metric("Posts shown", f"{len(df):,}")

# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if PAGE == "Overview":
    st.title("EchoAI – Overview")
    st.caption("Studying polarization and echo chambers in AI-only social networks (Chirper.AI)")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Posts",   f"{len(df_full):,}")
    c2.metric("Agents",        f"{df_full['agent_id'].nunique():,}")
    c3.metric("Extremist %",   f"{(df_full['agent_type']=='extremist').mean()*100:.1f}%")
    if 'sentiment_compound' in df_full.columns:
        c4.metric("Avg VADER",     f"{df_full['sentiment_compound'].mean():.3f}")
    if 'bert_score' in df_full.columns:
        c5.metric("Avg BERT Score",f"{df_full['bert_score'].mean():.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Agent Type Distribution")
        ac = df['agent_type'].value_counts().reset_index()
        ac.columns = ['agent_type', 'count']
        fig = px.pie(ac, names='agent_type', values='count',
                     color='agent_type', color_discrete_map=AGENT_COLORS,
                     hole=0.4)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=20, b=0))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Sentiment Distribution by Agent Type")
        if 'sentiment_label' in df.columns:
            sc = df.groupby(['agent_type', 'sentiment_label']).size().reset_index(name='count')
            fig2 = px.bar(sc, x='agent_type', y='count', color='sentiment_label',
                          color_discrete_map=SENT_COLORS, barmode='stack',
                          category_orders={'agent_type': list(AGENT_COLORS.keys())})
            fig2.update_layout(margin=dict(t=20, b=0), xaxis_title='', yaxis_title='Posts')
            st.plotly_chart(fig2, width='stretch')

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Engagement by Agent Type")
        fig3 = px.box(df, x='agent_type', y='engagement_score',
                      color='agent_type', color_discrete_map=AGENT_COLORS,
                      category_orders={'agent_type': list(AGENT_COLORS.keys())})
        fig3.update_layout(showlegend=False, xaxis_title='', margin=dict(t=20))
        st.plotly_chart(fig3, width='stretch')

    with col4:
        st.subheader("Posting Volume Over Time")
        if df['timestamp'].notna().any():
            td = df.groupby([df['timestamp'].dt.to_period('D').astype(str), 'agent_type'])\
                   .size().reset_index(name='count')
            td.columns = ['date', 'agent_type', 'count']
            fig4 = px.area(td, x='date', y='count', color='agent_type',
                           color_discrete_map=AGENT_COLORS)
            fig4.update_layout(xaxis_title='', margin=dict(t=20))
            st.plotly_chart(fig4, width='stretch')

    # Summary stats table
    st.subheader("Summary Statistics by Agent Type")
    cols_stat = [c for c in ['sentiment_compound', 'bert_score', 'engagement_score',
                              'likes', 'shares', 'word_count'] if c in df.columns]
    st.dataframe(
        df.groupby('agent_type')[cols_stat].mean().round(3).style.background_gradient(cmap='RdYlGn', axis=0),
        width='stretch',
    )

# ══════════════════════════════════════════════════════════════
# PAGE: CLASSIFIER (merged from api.py — text classification UI)
# ══════════════════════════════════════════════════════════════
elif PAGE == "Classifier":
    st.title("🎯 Text Classifier")
    st.caption("Type any text below — the Random Forest model will predict which AI agent type likely wrote it.")

    CATEGORY_META = {
        "extremist":    {"emoji": "🔴", "color": "#ef4444", "desc": "Content promoting radical or violent ideologies"},
        "conservative": {"emoji": "🟠", "color": "#f97316", "desc": "Content reflecting traditional or right-leaning views"},
        "moderate":     {"emoji": "🟡", "color": "#eab308", "desc": "Content with balanced, centrist perspectives"},
        "neutral":      {"emoji": "⚪", "color": "#94a3b8", "desc": "Objective content without political leaning"},
        "progressive":  {"emoji": "🔵", "color": "#3b82f6", "desc": "Content reflecting reform-oriented or left-leaning views"},
    }
    DEFAULTS = {'bert_score': 0.68, 'likes': 250.6, 'shares': 49.6, 'engagement_score': 300.2}

    rf, tfidf, le, scaler_m, hand_cols, tfidf_sim, vader, stop_w, df_samp, sim_matrix = load_classifier()

    if rf is None:
        st.error("Classifier model files missing in `models/`. Run `pipeline.py` first.")
        st.stop()

    def preprocess_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        return ' '.join([t for t in tokens if t not in stop_w and len(t) > 2])

    def extract_features(text, cleaned):
        X_tfidf = tfidf.transform([cleaned])
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
        X_hand   = csr_matrix(scaler_m.transform(raw_hand))
        return hstack([X_tfidf, X_hand])

    # ── Example buttons ──
    examples = {
        "Extremist": "We must destroy the system and burn it all to the ground. No mercy for our enemies.",
        "Progressive": "Climate justice and indigenous rights are essential for a sustainable future. Together we can transform society.",
        "Conservative": "Traditional values and law and order must be protected. We need stronger borders and family principles.",
        "Moderate": "Both sides have valid arguments. Let's debate the ethics and find a balanced solution through dialogue.",
        "Neutral": "The algorithm processes data through a neural network to generate the output classification.",
    }

    col_ex = st.columns(5)
    for i, (label, txt) in enumerate(examples.items()):
        if col_ex[i].button(label, width='stretch'):
            st.session_state['classifier_text'] = txt

    user_text = st.text_area(
        "Enter text to classify:",
        value=st.session_state.get('classifier_text', ''),
        height=120,
        placeholder="Type or paste a social media post here…",
    )

    if st.button("🚀 Classify Text", type="primary", width='stretch'):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing…"):
                cleaned = preprocess_text(user_text)
                features = extract_features(user_text, cleaned)

                pred       = rf.predict(features)[0]
                pred_label = le.inverse_transform([pred])[0].lower()
                probs      = rf.predict_proba(features)[0]
                prob_dict  = {cls.lower(): round(float(p) * 100, 1)
                              for cls, p in zip(le.classes_, probs)}
                vs         = vader.polarity_scores(user_text)
                meta       = CATEGORY_META.get(pred_label, {})

            # ── Result hero ──
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg, {meta.get('color','#666')}22, transparent);
                            border:1px solid {meta.get('color','#666')};
                            border-radius:16px; padding:30px; margin:20px 0; text-align:center;">
                    <div style="font-size:64px;">{meta.get('emoji','?')}</div>
                    <h2 style="color:{meta.get('color','#fff')}; margin:10px 0; text-transform:capitalize;">
                        {pred_label}
                    </h2>
                    <p style="color:#94a3b8;">{meta.get('desc','')}</p>
                    <div style="font-size:32px; font-weight:700; color:{meta.get('color','#fff')};">
                        {prob_dict[pred_label]}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Probability bars ──
            st.subheader("Probability Breakdown")
            prob_df = pd.DataFrame([
                {"Agent": k.title(), "Probability": v, "color": CATEGORY_META.get(k, {}).get('color', '#666')}
                for k, v in sorted(prob_dict.items(), key=lambda x: -x[1])
            ])
            fig = px.bar(
                prob_df, x="Probability", y="Agent", orientation='h',
                color="Agent", color_discrete_map={k.title(): v['color'] for k, v in CATEGORY_META.items()},
                text="Probability",
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='', xaxis_range=[0, 100],
                              height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width='stretch')

            # ── VADER details ──
            with st.expander("VADER Sentiment Details"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Compound", f"{vs['compound']:.3f}")
                c2.metric("Positive", f"{vs['pos']:.3f}")
                c3.metric("Negative", f"{vs['neg']:.3f}")
                c4.metric("Neutral",  f"{vs['neu']:.3f}")
                st.caption(f"Cleaned text used by TF-IDF: `{cleaned or '(empty after cleaning)'}`")

            # ── Similar posts ──
            if df_samp is not None and sim_matrix is not None:
                sim_vec    = tfidf_sim.transform([cleaned])
                sim_scores = cosine_similarity(sim_vec, sim_matrix)[0]
                top_idx    = np.argsort(sim_scores)[-5:][::-1]
                similar = [(df_samp.iloc[i], sim_scores[i]) for i in top_idx if sim_scores[i] > 0.05]

                if similar:
                    st.subheader("Similar Posts from Dataset")
                    for row, score in similar:
                        at = row['agent_type'].lower()
                        col_meta = CATEGORY_META.get(at, {})
                        st.markdown(
                            f"""
                            <div style="border-left:4px solid {col_meta.get('color','#666')};
                                        padding:12px 16px; margin:8px 0; background:rgba(255,255,255,0.02);
                                        border-radius:6px;">
                                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                    <span style="color:{col_meta.get('color','#fff')}; font-weight:600;">
                                        {col_meta.get('emoji','')} {at.title()}
                                    </span>
                                    <span style="color:#94a3b8; font-size:12px;">
                                        {score*100:.1f}% similar
                                    </span>
                                </div>
                                <div style="color:#cbd5e1; font-size:13px;">{row['text'][:250]}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ══════════════════════════════════════════════════════════════
# PAGE: SENTIMENT
# ══════════════════════════════════════════════════════════════
elif PAGE == "Sentiment":
    st.title("Sentiment Analysis")

    if 'sentiment_compound' not in df.columns:
        st.error("Sentiment columns not found. Run pipeline.py first.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean VADER",      f"{df['sentiment_compound'].mean():.3f}")
    c2.metric("Mean BERT Score", f"{df['bert_score'].mean():.3f}" if 'bert_score' in df.columns else "N/A")
    agree = ((df['sentiment_label'] == 'positive') & (df['bert_label'] == 'positive') |
             (df['sentiment_label'] == 'negative') & (df['bert_label'] == 'negative') |
             (df['sentiment_label'] == 'neutral')  & (df['bert_label'] == 'neutral')
             ).mean() if 'bert_label' in df.columns else 0
    c3.metric("VADER/BERT Agreement", f"{agree*100:.1f}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("VADER vs BERT Label Comparison")
        if 'bert_label' in df.columns:
            cross = pd.crosstab(df['sentiment_label'], df['bert_label'])
            fig = px.imshow(cross, text_auto=True, color_continuous_scale='Blues',
                            labels={'color': 'Count'})
            fig.update_layout(xaxis_title='BERT Label', yaxis_title='VADER Label')
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Sentiment Score Distribution")
        fig2 = px.histogram(df, x='sentiment_compound', color='agent_type',
                            nbins=40, color_discrete_map=AGENT_COLORS, opacity=0.7,
                            barmode='overlay')
        fig2.update_layout(xaxis_title='VADER Compound Score', yaxis_title='Count')
        st.plotly_chart(fig2, width='stretch')

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("VADER Sentiment by Agent Type")
        sc_v = df.groupby(['agent_type', 'sentiment_label']).size().reset_index(name='count')
        fig3 = px.bar(sc_v, x='agent_type', y='count', color='sentiment_label',
                      color_discrete_map=SENT_COLORS, barmode='group',
                      category_orders={'agent_type': list(AGENT_COLORS.keys())})
        fig3.update_layout(xaxis_title='', margin=dict(t=20))
        st.plotly_chart(fig3, width='stretch')

    with col4:
        st.subheader("BERT Label by Agent Type")
        if 'bert_label' in df.columns:
            sc_b = df.groupby(['agent_type', 'bert_label']).size().reset_index(name='count')
            fig4 = px.bar(sc_b, x='agent_type', y='count', color='bert_label',
                          color_discrete_map=SENT_COLORS, barmode='group',
                          category_orders={'agent_type': list(AGENT_COLORS.keys())})
            fig4.update_layout(xaxis_title='', margin=dict(t=20))
            st.plotly_chart(fig4, width='stretch')

    st.divider()
    st.subheader("Sentiment Score Boxplot by Agent Type")
    metric_choice = st.radio("Score type", ["sentiment_compound", "bert_score"],
                              horizontal=True, label_visibility="collapsed")
    if metric_choice in df.columns:
        fig5 = px.violin(df, x='agent_type', y=metric_choice, color='agent_type',
                         box=True, color_discrete_map=AGENT_COLORS,
                         category_orders={'agent_type': list(AGENT_COLORS.keys())})
        fig5.update_layout(showlegend=False, xaxis_title='')
        st.plotly_chart(fig5, width='stretch')

# ══════════════════════════════════════════════════════════════
# PAGE: TOPICS
# ══════════════════════════════════════════════════════════════
elif PAGE == "Topics":
    st.title("LDA Topic Modelling")

    if 'dominant_topic' not in df.columns:
        st.warning("Topic columns not found. Run pipeline.py first.")
        if os.path.exists('figures/lda_topics.png'):
            st.image('figures/lda_topics.png')
        st.stop()

    lda, dictionary, topic_words_dict = load_lda()
    NUM_TOPICS = df['dominant_topic'].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Topics", NUM_TOPICS)
    c2.metric("Avg Topic Confidence", f"{df['topic_confidence'].mean():.3f}" if 'topic_confidence' in df.columns else "N/A")
    c3.metric("Posts Covered", f"{len(df):,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Posts per Topic")
        tc = df['dominant_topic'].value_counts().sort_index().reset_index()
        tc.columns = ['topic', 'count']
        fig = px.bar(tc, x='topic', y='count', color='topic',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, xaxis_title='Topic', yaxis_title='Posts')
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Topic × Agent Type Heatmap")
        cross = pd.crosstab(df['dominant_topic'], df['agent_type'])
        fig2 = px.imshow(cross, text_auto=True, color_continuous_scale='Blues',
                         labels={'color': 'Posts', 'x': 'Agent Type', 'y': 'Topic'})
        st.plotly_chart(fig2, width='stretch')

    st.divider()
    st.subheader("Topic Word Clouds")
    if os.path.exists('figures/lda_wordclouds.png'):
        st.image('figures/lda_wordclouds.png', width='stretch')

    st.divider()
    st.subheader("Topic Explorer")
    if topic_words_dict:
        sel_topic = st.selectbox("Select Topic", list(topic_words_dict.keys()),
                                 format_func=lambda x: f"Topic {x}")
        words = topic_words_dict.get(sel_topic, [])
        st.write(f"**Top words for Topic {sel_topic}:**")
        st.write("  ·  ".join(f"`{w}`" for w in words[:15]))

        st.subheader(f"Sample posts – Topic {sel_topic}")
        samp = df[df['dominant_topic'] == sel_topic][['text', 'agent_type', 'topic_confidence']]\
            .dropna().sort_values('topic_confidence', ascending=False).head(5)
        for _, row in samp.iterrows():
            with st.expander(f"{row['agent_type'].title()} — confidence: {row['topic_confidence']:.2f}"):
                st.write(row['text'])

    if 'topic_confidence' in df.columns:
        st.subheader("Topic Confidence Distribution")
        fig3 = px.box(df, x='dominant_topic', y='topic_confidence',
                      color_discrete_sequence=['#3b82f6'])
        fig3.update_layout(xaxis_title='Topic', yaxis_title='Confidence')
        st.plotly_chart(fig3, width='stretch')

# ══════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════════
elif PAGE == "Clustering":
    st.title("Clustering Analysis")

    if 'kmeans_cluster' not in df.columns:
        st.warning("Cluster columns not found. Run pipeline.py first.")
        if os.path.exists('figures/clustering_pca.png'):
            st.image('figures/clustering_pca.png')
        st.stop()

    n_clusters = df['kmeans_cluster'].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("K-Means Clusters", n_clusters)
    c2.metric("Posts Clustered", f"{(df['kmeans_cluster'] >= 0).sum():,}")
    if 'hier_cluster' in df.columns:
        c3.metric("Hier. Sample Size", f"{(df['hier_cluster'] >= 0).sum():,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PCA – Colored by Cluster")
        if 'pca_x' in df.columns:
            fig = px.scatter(df, x='pca_x', y='pca_y', color=df['kmeans_cluster'].astype(str),
                             opacity=0.5, labels={'color': 'Cluster'},
                             color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_traces(marker_size=3)
            fig.update_layout(margin=dict(t=20))
            st.plotly_chart(fig, width='stretch')
        elif os.path.exists('figures/clustering_pca.png'):
            st.image('figures/clustering_pca.png')

    with col2:
        st.subheader("PCA – Colored by Agent Type")
        if 'pca_x' in df.columns:
            fig2 = px.scatter(df, x='pca_x', y='pca_y', color='agent_type',
                              color_discrete_map=AGENT_COLORS, opacity=0.5)
            fig2.update_traces(marker_size=3)
            fig2.update_layout(margin=dict(t=20))
            st.plotly_chart(fig2, width='stretch')

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Cluster Size Distribution")
        cc = df['kmeans_cluster'].value_counts().sort_index().reset_index()
        cc.columns = ['cluster', 'count']
        fig3 = px.bar(cc, x='cluster', y='count',
                      color_discrete_sequence=['#3b82f6'])
        fig3.update_layout(xaxis_title='Cluster', yaxis_title='Posts')
        st.plotly_chart(fig3, width='stretch')

    with col4:
        st.subheader("Cluster × Agent Type")
        cross = pd.crosstab(df['kmeans_cluster'], df['agent_type'])
        fig4 = px.imshow(cross, text_auto=True, color_continuous_scale='Viridis',
                         labels={'color': 'Posts', 'x': 'Agent Type', 'y': 'Cluster'})
        st.plotly_chart(fig4, width='stretch')

    st.divider()
    st.subheader("Cluster Statistics")
    stat_cols = [c for c in ['sentiment_compound', 'bert_score', 'engagement_score',
                              'likes', 'shares', 'word_count', 'hashtag_count'] if c in df.columns]
    st.dataframe(
        df.groupby('kmeans_cluster')[stat_cols].mean().round(3).style.background_gradient(cmap='coolwarm', axis=0),
        width='stretch',
    )

    if os.path.exists('figures/clustering_elbow.png'):
        with st.expander("Show Elbow / Silhouette Plot"):
            st.image('figures/clustering_elbow.png', width='stretch')

    if os.path.exists('figures/dendrogram.png'):
        with st.expander("Show Dendrogram (200-post sample)"):
            st.image('figures/dendrogram.png', width='stretch')

# ══════════════════════════════════════════════════════════════
# PAGE: NETWORK
# ══════════════════════════════════════════════════════════════
elif PAGE == "Network":
    st.title("Network Analysis")

    if cent_df is None:
        st.warning("Network data not found. Run pipeline.py first.")
        if os.path.exists('figures/network_graph.png'):
            st.image('figures/network_graph.png')
        st.stop()

    # count edges
    n_edges = 0
    if os.path.exists('models/network_graph.pkl'):
        try:
            with open('models/network_graph.pkl', 'rb') as f:
                net_data = pickle.load(f)
            G = net_data['G']
            n_edges = G.number_of_edges()
        except Exception:
            G = None
    else:
        G = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents (Nodes)",  f"{len(cent_df):,}")
    c2.metric("Interactions (Edges)", f"{n_edges:,}")
    c3.metric("Avg PageRank",    f"{cent_df['pagerank'].mean():.4f}")
    c4.metric("Avg Betweenness", f"{cent_df['betweenness'].mean():.4f}")

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Network Graph")
        if os.path.exists('network_interactive.html'):
            with open('network_interactive.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            import streamlit.components.v1 as components
            components.html(html_content, height=550, scrolling=True)
        elif os.path.exists('figures/network_graph.png'):
            st.image('figures/network_graph.png', width='stretch')

    with col2:
        st.subheader("Top 15 Influential Agents")
        top15 = cent_df.head(15)[['agent_name', 'agent_type', 'betweenness', 'pagerank', 'in_degree']]
        st.dataframe(top15.style.background_gradient(subset=['betweenness', 'pagerank'], cmap='Oranges'),
                     width='stretch')

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("PageRank by Agent Type")
        fig = px.box(cent_df, x='agent_type', y='pagerank',
                     color='agent_type', color_discrete_map=AGENT_COLORS)
        fig.update_layout(showlegend=False, xaxis_title='')
        st.plotly_chart(fig, width='stretch')

    with col4:
        st.subheader("Betweenness Centrality by Agent Type")
        fig2 = px.box(cent_df, x='agent_type', y='betweenness',
                      color='agent_type', color_discrete_map=AGENT_COLORS)
        fig2.update_layout(showlegend=False, xaxis_title='')
        st.plotly_chart(fig2, width='stretch')

    if os.path.exists('figures/centrality_boxplot.png'):
        with st.expander("Show Centrality Boxplot (all metrics)"):
            st.image('figures/centrality_boxplot.png', width='stretch')

    st.subheader("Full Centrality Table")
    sort_by = st.selectbox("Sort by", ['betweenness', 'pagerank', 'degree', 'in_degree', 'out_degree'])
    st.dataframe(cent_df.sort_values(sort_by, ascending=False).reset_index(drop=True),
                 width='stretch')

# ══════════════════════════════════════════════════════════════
# PAGE: POLARIZATION
# ══════════════════════════════════════════════════════════════
elif PAGE == "Polarization":
    st.title("Polarization Trends")

    if daily_df is None:
        st.warning("Daily trend data not found. Run pipeline.py first.")
        if os.path.exists('figures/timeseries_trends.png'):
            st.image('figures/timeseries_trends.png')
        st.stop()

    try:
        import pymannkendall as mk
        pol_result = mk.original_test(daily_df['polarization_ratio'].dropna())
        vader_result = mk.original_test(daily_df['mean_vader'].dropna())
        mk_available = True
    except Exception:
        mk_available = False

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days Tracked",        f"{len(daily_df):,}")
    c2.metric("Avg Polarization",    f"{daily_df['polarization_ratio'].mean():.3f}")
    c3.metric("Avg VADER",           f"{daily_df['mean_vader'].mean():.3f}")
    if mk_available:
        c4.metric("Pol. Trend",      pol_result.trend.title(),
                  delta=f"p={pol_result.p:.3f}")

    if mk_available:
        st.info(
            f"**Mann-Kendall Test Results** – "
            f"Polarization: `{pol_result.trend}` (p={pol_result.p:.4f}, τ={pol_result.Tau:.3f})  |  "
            f"VADER Sentiment: `{vader_result.trend}` (p={vader_result.p:.4f}, τ={vader_result.Tau:.3f})"
        )

    st.divider()

    st.subheader("Sentiment & Polarization Over Time")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['vader_ma3'],
                             name='VADER (3-day MA)', line=dict(color='#3b82f6', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['bert_ma3'],
                             name='BERT (3-day MA)', line=dict(color='#22c55e', width=2, dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['polar_ma3'],
                             name='Polarization (3-day MA)', line=dict(color='#ef4444', width=2)), secondary_y=True)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
    fig.update_yaxes(title_text="Polarization Ratio", secondary_y=True)
    fig.update_layout(height=400, margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily Posting Volume by Agent Type")
        if os.path.exists('figures/timeseries_stacked.png'):
            st.image('figures/timeseries_stacked.png', width='stretch')

    with col2:
        st.subheader("Activity Heatmap")
        if os.path.exists('figures/timeseries_heatmap.png'):
            st.image('figures/timeseries_heatmap.png', width='stretch')

    st.divider()
    st.subheader("Polarization Ratio Distribution")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=daily_df['polarization_ratio'], nbinsx=20,
                                marker_color='#ef4444', opacity=0.7, name='Polarization Ratio'))
    fig2.update_layout(xaxis_title='Daily Polarization Ratio', yaxis_title='Days',
                       showlegend=False, height=300)
    st.plotly_chart(fig2, width='stretch')

    with st.expander("Show Raw Daily Trends Data"):
        st.dataframe(daily_df, width='stretch')

# ══════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif PAGE == "Data Explorer":
    st.title("Data Explorer")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Columns",    f"{len(df.columns)}")
    c3.metric("Agent Types", f"{df['agent_type'].nunique()}")

    st.divider()

    # Column selector
    all_cols = df.columns.tolist()
    default_cols = [c for c in ['post_id', 'agent_name', 'agent_type', 'text',
                                  'sentiment_label', 'bert_label', 'dominant_topic',
                                  'kmeans_cluster', 'engagement_score', 'timestamp']
                    if c in all_cols]
    sel_cols = st.multiselect("Columns to display", all_cols, default=default_cols)

    if sel_cols:
        st.dataframe(df[sel_cols].reset_index(drop=True), width='stretch', height=500)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download filtered CSV",
            data=csv_data,
            file_name="echoai_filtered.csv",
            mime="text/csv",
        )
    with col2:
        st.caption(f"Dataset: `{DATA_FILE if os.path.exists(DATA_FILE) else FALLBACK}`")
        st.caption(f"Columns: {', '.join(df.columns[:10].tolist())}…")

    with st.expander("Column descriptions"):
        col_desc = {
            'post_id': 'Unique post identifier',
            'agent_id': 'Unique agent identifier',
            'agent_type': 'One of: extremist, conservative, moderate, neutral, progressive',
            'text': 'Raw post text',
            'cleaned_text': 'Preprocessed text (lowercase, stopwords removed)',
            'sentiment_compound': 'VADER compound score (-1 to +1)',
            'sentiment_label': 'VADER label: positive/negative/neutral',
            'bert_label': 'BERT-based label: positive/negative/neutral',
            'bert_score': 'BERT confidence score (0-1)',
            'dominant_topic': 'LDA dominant topic ID (0-4)',
            'topic_confidence': 'LDA topic assignment confidence',
            'kmeans_cluster': 'K-Means cluster ID',
            'hier_cluster': 'Hierarchical cluster ID (200-post sample, -1 = not sampled)',
            'pca_x': 'PCA dimension 1 for visualization',
            'pca_y': 'PCA dimension 2 for visualization',
            'is_polarized': '1 = extremist post, 0 = otherwise',
            'reply_to': 'post_id of the post being replied to (NaN = original post)',
        }
        for col, desc in col_desc.items():
            if col in df.columns:
                st.write(f"**`{col}`** – {desc}")

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption("EchoAI Research Dashboard · MSc Data Science · Chirper.AI Polarization Study")
