import pandas as pd
import pickle
import numpy as np
import random
from collections import Counter
import math
import re

from models import LogisticRegressionManual, TfidfVectorizerManual

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    test_size = int(test_size * n_samples)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def accuracy_score_manual(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

# Load dữ liệu
df = pd.read_csv("IMDB Dataset.csv")

# Tiền xử lý dữ liệu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # xóa thẻ HTML
    text = re.sub(r'[^a-z\s]', '', text)  # xóa ký tự đặc biệt
    return text

df['review'] = df['review'].apply(preprocess_text)

# Vectorize văn bản
vectorizer = TfidfVectorizerManual(max_features=5000, stop_words=['the', 'is', 'and', 'to', 'a'])
X = vectorizer.fit_transform(df['review'].values)

# Nhãn
y = df['sentiment'].map({'positive': 1, 'negative': 0}).values

# Tách train-test
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LogisticRegressionManual(learning_rate=0.1, max_iter=100)
model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score_manual(y_test, y_pred))

# Lưu mô hình và vectorizer
with open('nlp_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)