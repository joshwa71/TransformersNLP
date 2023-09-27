from transformers import pipeline

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

classifier = pipeline('sentiment-analysis', device=0)

df_ = pd.read_csv('data/AirlineTweets.csv')
df = df_[['airline_sentiment', 'text']].copy()
df = df[df['airline_sentiment'] != 'neutral'].copy()

target_map = {'negative': 0, 'positive': 1}
df['target'] = df['airline_sentiment'].map(target_map)

texts = df['text'].to_list()
predictions = classifier(texts)

probs = [d["score"] if d['label'].startswith('P') else 1 - d["score"] for d in predictions]
preds = [1 if d['label'].startswith('P') else 0 for d in predictions]

preds = np.array(preds)

print("accuracy:", np.mean(df['target'] == preds))

cm = confusion_matrix(df['target'], preds, normalize='true')











