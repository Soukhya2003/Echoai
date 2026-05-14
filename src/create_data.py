import pandas as pd
import joblib

df = pd.read_excel("EchoAI_Chirper_dataset.xlsx")
print("Columns in dataframe:", df.columns.tolist())

# Use the correct column name - adjust based on actual name
# If 'cleaned_text' doesn't exist, try 'text' or whatever you used for TF-IDF
text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
df_sample = df[['text', text_col, 'agent_type']].dropna().head(2000)
df_sample.to_csv("sample_data.csv", index=False)

tfidf = joblib.load("tfidf.pkl")
tfidf_matrix_sample = tfidf.transform(df_sample[text_col])
joblib.dump(tfidf_matrix_sample, "tfidf_matrix_sample.pkl")

print("✅ Files created: sample_data.csv and tfidf_matrix_sample.pkl")