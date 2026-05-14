import pandas as pd
import joblib

print("Loading Excel file...")
df = pd.read_excel("EchoAI_Chirper_dataset.xlsx")

print("Creating sample_data.csv...")
df_sample = df[['text', 'cleaned_text', 'agent_type']].dropna().head(2000)
df_sample.to_csv("sample_data.csv", index=False)

print("Loading tfidf.pkl...")
tfidf = joblib.load("tfidf.pkl")

print("Creating tfidf_matrix_sample.pkl...")
tfidf_matrix_sample = tfidf.transform(df_sample['cleaned_text'])
joblib.dump(tfidf_matrix_sample, "tfidf_matrix_sample.pkl")

print("✅ Done! Files created: sample_data.csv , tfidf_matrix_sample.pkl")