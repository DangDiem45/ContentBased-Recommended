from collections import Counter
import math
import numpy as np

class TfidfVectorizerManual:
    def __init__(self, max_features=5000, stop_words=None):
        self.max_features = max_features
        self.stop_words = stop_words if stop_words else []
        self.vocab = {}
        self.idf = {}
        
    def fit_transform(self, texts):
        # Tạo từ điển từ vựng
        word_counts = Counter()
        for text in texts:
            words = set(text.split())
            word_counts.update(words)
        
        # Lấy top max_features từ phổ biến nhất
        self.vocab = {
            word: idx for idx, (word, _) in enumerate(
                [item for item in word_counts.most_common(self.max_features) if item[0] not in self.stop_words]
            )
        }
        
        # Tính IDF
        n_docs = len(texts)
        doc_freq = Counter()
        for text in texts:
            words = set(text.split())
            doc_freq.update(words)
        
        for word in self.vocab:
            self.idf[word] = math.log(n_docs / (1 + doc_freq[word]))
        
        # Tính TF-IDF
        X = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            word_count = Counter(text.split())
            for word, count in word_count.items():
                if word in self.vocab:
                    tf = count / len(text.split())
                    X[i, self.vocab[word]] = tf * self.idf[word]
        
        return X
    
    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            word_count = Counter(text.split())
            for word, count in word_count.items():
                if word in self.vocab:
                    tf = count / len(text.split())
                    X[i, self.vocab[word]] = tf * self.idf[word]
        return X
    
class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Tính dự đoán
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Tính gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Cập nhật trọng số
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)
    
def cosine_similarity_manual(X):
    """
    Tính ma trận độ tương đồng cosine giữa tất cả các vector trong X.
    X: Ma trận đầu vào (n_samples, n_features), có thể là ma trận thưa hoặc đầy đủ.
    Trả về: Ma trận cosine similarity (n_samples, n_samples).
    """
    # Tính độ dài Euclidean của mỗi vector
    norms = np.sqrt(np.sum(X * X, axis=1))
    norms[norms == 0] = 1  # Tránh chia cho 0
    normalized_X = X / norms[:, np.newaxis]  # Chuẩn hóa các vector
    
    # Tính tích vô hướng giữa các vector
    if hasattr(X, "toarray"):  # Nếu X là ma trận thưa
        X = X.toarray()
    sim_matrix = np.dot(X, X.T)  # Ma trận tương đồng cosine
    
    return sim_matrix