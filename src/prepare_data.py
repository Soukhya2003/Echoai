import pandas as pd
import joblib

df = pd.read_excel("EchoAI_Chirper_dataset_BERT.xlsx")
df_sample = df[['text', 'cleaned_text', 'agent_type']].dropna().head(2000)
df_sample.to_csv("sample_data.csv", index=False)

tfidf = joblib.load("tfidf.pkl")
tfidf_matrix_sample = tfidf.transform(df_sample['cleaned_text'])
joblib.dump(tfidf_matrix_sample, "tfidf_matrix_sample.pkl")

print("✅ Sample data prepared!")
