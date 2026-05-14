import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import numpy as np

print("Loading data...")
df = pd.read_excel("EchoAI_Chirper_dataset_BERT.xlsx")

le = joblib.load("label_encoder.pkl")
labels = le.transform(df['agent_type'])
texts = df['cleaned_text'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

train_enc = tokenize_function(train_texts)
val_enc = tokenize_function(val_texts)

class AgentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = AgentDataset(train_enc, train_labels)
val_dataset = AgentDataset(val_enc, val_labels)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

training_args = TrainingArguments(
    output_dir='./bert_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")
joblib.dump(le, "label_encoder_bert.pkl")

print("Done! Model saved.")