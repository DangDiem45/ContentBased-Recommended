import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

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
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['review'])

# Nhãn
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Tách train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LogisticRegression()
model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Lưu mô hình và vectorizer
with open('nlp_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizerer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
