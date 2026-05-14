# EchoAI – Polarization & Echo Chambers in AI-Only Social Networks

An MSc Data Science research project analysing whether AI agents on AI-only social networks
(Chirper.AI) develop polarised opinions and form echo chambers.

## Research Questions

1. Do AI agents cluster into ideologically homogeneous communities (echo chambers)?
2. Is there a measurable trend in polarisation over time?
3. Which agents act as bridges or amplifiers across ideological lines?
4. Can machine learning reliably detect polarised content from text alone?

## Dataset

| Property | Value |
|---|---|
| Total posts | 3,000 |
| Real posts (Chirper.AI) | 64 |
| Synthetic posts | 2,936 |
| Agent types | Extremist, Conservative, Moderate, Neutral, Progressive |
| Date range | Jan–Mar 2024 (synthetic timestamps) |

Columns include raw text, VADER sentiment scores, BERT sentiment scores, engagement metrics
(likes, shares), reply graph structure, and derived features.

## Pipeline Steps

| Step | Description | Output |
|---|---|---|
| 1–4 | Data collection, generation, preprocessing, VADER | `EchoAI_Chirper_dataset_BERT.xlsx` |
| 5 | BERT sentiment analysis (twitter-roberta-base) | `bert_label`, `bert_score` columns |
| 6 | LDA topic modelling (5 topics) | `dominant_topic`, `topic_confidence` |
| 7 | K-Means + hierarchical clustering | `kmeans_cluster`, `hier_cluster`, PCA coords |
| 8 | Random Forest classification (sentiment + polarisation) | `rf_model.pkl`, figures |
| 9 | Network analysis (NetworkX + Louvain communities) | `network_centrality.csv`, graph figures |
| 10 | Time series polarisation tracking (Mann-Kendall) | `daily_trends.csv`, trend figures |
| 11 | Streamlit dashboard (7 pages) | `app.py` |

## Setup

```bash
# Use Python 3.11 (required – data science packages not yet available for 3.12+)
py -3.11 -m pip install -r requirements.txt

# Download NLTK data
py -3.11 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Running the Project

### 1 – Run the analysis pipeline (Steps 6–10)
```bash
py -3.11 pipeline.py
```
This generates all model artefacts, figures, and the final dataset.  
Runtime: ~5–15 minutes depending on hardware.

### 2 – Launch the Streamlit dashboard
```bash
py -3.11 -m streamlit run app.py
```
Opens at http://localhost:8501

### 3 – Start the REST API (for the HTML classifier UI)
```bash
py -3.11 -m uvicorn api:app --reload
```
API available at http://localhost:8000  
Open `index.html` in a browser to use the classifier UI.

## Project Structure

```
Soukhya_project/
├── data/
│   ├── EchoAI_Chirper_dataset.xlsx          # Base dataset (21 cols)
│   ├── EchoAI_Chirper_dataset_BERT.xlsx     # + VADER & BERT scores (30 cols)
│   └── EchoAI_Chirper_dataset_FINAL.xlsx    # + topics, clusters, PCA (generated)
├── models/
│   ├── rf_model.pkl                          # RF classifier (agent type, 211 features)
│   ├── rf_sentiment.pkl                      # RF classifier (sentiment)
│   ├── rf_polarization.pkl                   # RF classifier (polarisation)
│   ├── tfidf.pkl                             # TF-IDF vectoriser
│   ├── label_encoder.pkl                     # Agent-type label encoder
│   ├── lda_model*                            # Gensim LDA model files
│   └── dictionary.gensim*                   # Gensim corpus dictionary
├── figures/                                  # All saved PNG figures
├── src/
│   ├── train_bert.py                         # BERT fine-tuning script
│   ├── train_simple_model.py                 # TF-IDF + RF training
│   ├── prepare_data.py                       # Rebuild sample CSV
│   └── create_data.py / create_sample_data.py
├── notebooks/
│   └── Echo AI Project (3).ipynb            # Exploratory analysis
├── app.py                                    # Streamlit dashboard (7 pages)
├── api.py                                    # FastAPI REST endpoint
├── pipeline.py                               # End-to-end Steps 6–10
├── index.html                                # Standalone HTML classifier UI
├── requirements.txt
└── README.md
```

## Key Outputs

| File | Description |
|---|---|
| `EchoAI_Chirper_dataset_FINAL.xlsx` | Final enriched dataset with all derived columns |
| `network_centrality.csv` | Per-agent centrality metrics (degree, betweenness, PageRank) |
| `daily_trends.csv` | Daily aggregated sentiment + polarisation ratios |
| `network_interactive.html` | Interactive network graph (pyvis) |
| `figures/` | 15+ analysis figures (PNGs) |

## Dashboard Pages

1. **Overview** – KPI cards, agent distribution, engagement stats
2. **Sentiment** – VADER vs BERT comparison, distributions by agent type
3. **Topics** – LDA topic explorer, word clouds, topic×agent heatmap
4. **Clustering** – Interactive PCA scatter, cluster stats, dendrogram
5. **Network** – Interactive agent graph, centrality leaderboard
6. **Polarization** – Time series trends, Mann-Kendall results, heatmap
7. **Data Explorer** – Searchable table with CSV download

## Technology Stack

- **ML/NLP**: scikit-learn, gensim (LDA), NLTK, VaderSentiment, Hugging Face Transformers
- **Network**: NetworkX, python-louvain (Louvain community detection)
- **Stats**: scipy, pymannkendall
- **Viz**: plotly, matplotlib, seaborn, wordcloud, pyvis
- **API/UI**: FastAPI, Streamlit
