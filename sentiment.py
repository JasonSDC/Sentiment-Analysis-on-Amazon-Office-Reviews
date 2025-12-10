import re
import csv
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gensim.downloader as api

warnings.filterwarnings('ignore')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_neural_network(X_train, y_train, X_test, y_test, input_dim, epochs=20, batch_size=32, lr=0.001):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = FeedForwardNN(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        _, train_preds = torch.max(train_outputs, 1)
        train_preds = train_preds.numpy()
        
        test_outputs = model(X_test_tensor)
        _, test_preds = torch.max(test_outputs, 1)
        test_preds = test_preds.numpy()
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, train_preds),
        'precision': precision_score(y_train, train_preds, average='binary'),
        'recall': recall_score(y_train, train_preds, average='binary'),
        'f1': f1_score(y_train, train_preds, average='binary')
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, test_preds),
        'precision': precision_score(y_test, test_preds, average='binary'),
        'recall': recall_score(y_test, test_preds, average='binary'),
        'f1': f1_score(y_test, test_preds, average='binary')
    }
    
    return train_metrics, test_metrics

def main():
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    
    import zipfile
    with zipfile.ZipFile('amazon_reviews_us_Office_Products_v1_00.tsv.zip', 'r') as z:
        df = pd.read_csv(z.open('amazon_reviews_us_Office_Products_v1_00.tsv'), sep='\t', quoting=csv.QUOTE_NONE)
    
    df = df.rename(columns={'review_body': 'review', 'star_rating': 'rating'})
    
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    positive_count = (df["rating"] > 3).sum()
    negative_count = (df["rating"] < 3).sum()
    neutral_count  = (df["rating"] == 3).sum()
    
    print(f"Positive reviews: {positive_count}")
    print(f"Negative reviews: {negative_count}")
    print(f"Neutral reviews: {neutral_count}")
    
    df_binary = df[df['rating'] != 3].copy()
    df_binary['label'] = (df_binary['rating'] > 3).astype(int)
    
    df_positive = df_binary[df_binary['label'] == 1]
    df_negative = df_binary[df_binary['label'] == 0]
    
    df_positive_sampled = df_positive.sample(n=100000, random_state=RANDOM_STATE)
    df_negative_sampled = df_negative.sample(n=100000, random_state=RANDOM_STATE)
    
    df_balanced = pd.concat([df_positive_sampled, df_negative_sampled])
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_balanced['review'].values,
        df_balanced['label'].values,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_balanced['label']
    )
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    avg_before = np.mean([len(str(review)) for review in X_train])
    
    X_train_cleaned = [clean_text(review) for review in X_train]
    avg_after = np.mean([len(review) for review in X_train_cleaned])
    
    print(f"Average length before cleaning: {avg_before:.4f}")
    print(f"Average length after cleaning: {avg_after:.4f}")
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(text):
        text = clean_text(str(text))
        
        words = text.split()
        
        words = [word for word in words if word not in stop_words]
        
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    X_train_preprocessed = [preprocess_text(review) for review in X_train]
    X_test_preprocessed = [preprocess_text(review) for review in X_test]
    
    avg_after_preprocessing = np.mean([len(text) for text in X_train_preprocessed])
    print(f"Average length after preprocessing: {avg_after_preprocessing:.4f}")
    
    tfidf = TfidfVectorizer(min_df=2, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train_preprocessed)
    X_test_tfidf = tfidf.transform(X_test_preprocessed)
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    perceptron = Perceptron(random_state=RANDOM_STATE)
    perceptron.fit(X_train_tfidf, y_train)
    
    y_train_pred_perc = perceptron.predict(X_train_tfidf)
    train_acc_perc = accuracy_score(y_train, y_train_pred_perc)
    train_prec_perc = precision_score(y_train, y_train_pred_perc, average='binary')
    train_rec_perc = recall_score(y_train, y_train_pred_perc, average='binary')
    train_f1_perc = f1_score(y_train, y_train_pred_perc, average='binary')
    
    y_test_pred_perc = perceptron.predict(X_test_tfidf)
    test_acc_perc = accuracy_score(y_test, y_test_pred_perc)
    test_prec_perc = precision_score(y_test, y_test_pred_perc, average='binary')
    test_rec_perc = recall_score(y_test, y_test_pred_perc, average='binary')
    test_f1_perc = f1_score(y_test, y_test_pred_perc, average='binary')
    
    print(f"Perceptron Training Accuracy: {train_acc_perc:.4f}")
    print(f"Perceptron Training Precision: {train_prec_perc:.4f}")
    print(f"Perceptron Training Recall: {train_rec_perc:.4f}")
    print(f"Perceptron Training F1-score: {train_f1_perc:.4f}")
    print(f"Perceptron Testing Accuracy: {test_acc_perc:.4f}")
    print(f"Perceptron Testing Precision: {test_prec_perc:.4f}")
    print(f"Perceptron Testing Recall: {test_rec_perc:.4f}")
    print(f"Perceptron Testing F1-score: {test_f1_perc:.4f}")
    
    svm = LinearSVC(random_state=RANDOM_STATE, max_iter=1000)
    svm.fit(X_train_tfidf, y_train)
    
    y_train_pred_svm = svm.predict(X_train_tfidf)
    train_acc_svm = accuracy_score(y_train, y_train_pred_svm)
    train_prec_svm = precision_score(y_train, y_train_pred_svm, average='binary')
    train_rec_svm = recall_score(y_train, y_train_pred_svm, average='binary')
    train_f1_svm = f1_score(y_train, y_train_pred_svm, average='binary')
    
    y_test_pred_svm = svm.predict(X_test_tfidf)
    test_acc_svm = accuracy_score(y_test, y_test_pred_svm)
    test_prec_svm = precision_score(y_test, y_test_pred_svm, average='binary')
    test_rec_svm = recall_score(y_test, y_test_pred_svm, average='binary')
    test_f1_svm = f1_score(y_test, y_test_pred_svm, average='binary')
    
    print(f"SVM Training Accuracy: {train_acc_svm:.4f}")
    print(f"SVM Training Precision: {train_prec_svm:.4f}")
    print(f"SVM Training Recall: {train_rec_svm:.4f}")
    print(f"SVM Training F1-score: {train_f1_svm:.4f}")
    print(f"SVM Testing Accuracy: {test_acc_svm:.4f}")
    print(f"SVM Testing Precision: {test_prec_svm:.4f}")
    print(f"SVM Testing Recall: {test_rec_svm:.4f}")
    print(f"SVM Testing F1-score: {test_f1_svm:.4f}")
    
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    
    y_train_pred_lr = lr.predict(X_train_tfidf)
    train_acc_lr = accuracy_score(y_train, y_train_pred_lr)
    train_prec_lr = precision_score(y_train, y_train_pred_lr, average='binary')
    train_rec_lr = recall_score(y_train, y_train_pred_lr, average='binary')
    train_f1_lr = f1_score(y_train, y_train_pred_lr, average='binary')
    
    y_test_pred_lr = lr.predict(X_test_tfidf)
    test_acc_lr = accuracy_score(y_test, y_test_pred_lr)
    test_prec_lr = precision_score(y_test, y_test_pred_lr, average='binary')
    test_rec_lr = recall_score(y_test, y_test_pred_lr, average='binary')
    test_f1_lr = f1_score(y_test, y_test_pred_lr, average='binary')
    
    print(f"Logistic Regression Training Accuracy: {train_acc_lr:.4f}")
    print(f"Logistic Regression Training Precision: {train_prec_lr:.4f}")
    print(f"Logistic Regression Training Recall: {train_rec_lr:.4f}")
    print(f"Logistic Regression Training F1-score: {train_f1_lr:.4f}")
    print(f"Logistic Regression Testing Accuracy: {test_acc_lr:.4f}")
    print(f"Logistic Regression Testing Precision: {test_prec_lr:.4f}")
    print(f"Logistic Regression Testing Recall: {test_rec_lr:.4f}")
    print(f"Logistic Regression Testing F1-score: {test_f1_lr:.4f}")
    
    glove_model = api.load('glove-wiki-gigaword-100')
    
    result = glove_model.most_similar(positive=['king', 'woman'], negative=['man'], topn=10)
    top_words = [word for word, _ in result[:5]]
    print("king - man + woman = ", end="")
    print(", ".join(top_words))
    
    result = glove_model.most_similar('outstanding', topn=5)
    print("Outstanding = ", end="")
    print(", ".join([word for word, _ in result]))
    
    def get_average_glove_vector(text, model, dim=100):
        words = text.split()
        vectors = []
        for word in words:
            if word in model:
                vectors.append(model[word])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(dim)
    
    X_train_glove_avg = np.array([get_average_glove_vector(text, glove_model) 
                                   for text in X_train_preprocessed])
    X_test_glove_avg = np.array([get_average_glove_vector(text, glove_model) 
                                  for text in X_test_preprocessed])
    
    train_metrics_avg, test_metrics_avg = train_neural_network(
        X_train_glove_avg, y_train, X_test_glove_avg, y_test, input_dim=100
    )
    
    print(f"Average Feature Training Accuracy: {train_metrics_avg['accuracy']:.4f}")
    print(f"Average Feature Training Precision: {train_metrics_avg['precision']:.4f}")
    print(f"Average Feature Training Recall: {train_metrics_avg['recall']:.4f}")
    print(f"Average Feature Training F1-score: {train_metrics_avg['f1']:.4f}")
    print(f"Average Feature Testing Accuracy: {test_metrics_avg['accuracy']:.4f}")
    print(f"Average Feature Testing Precision: {test_metrics_avg['precision']:.4f}")
    print(f"Average Feature Testing Recall: {test_metrics_avg['recall']:.4f}")
    print(f"Average Feature Testing F1-score: {test_metrics_avg['f1']:.4f}")
    
    def get_concatenated_glove_vectors(text, model, num_words=10, dim=100):
        words = text.split()[:num_words]
        vectors = []
        
        for i in range(num_words):
            if i < len(words) and words[i] in model:
                vectors.append(model[words[i]])
            else:
                vectors.append(np.zeros(dim))
        
        return np.concatenate(vectors)
    
    X_train_glove_concat = np.array([get_concatenated_glove_vectors(text, glove_model) 
                                      for text in X_train_preprocessed])
    X_test_glove_concat = np.array([get_concatenated_glove_vectors(text, glove_model) 
                                     for text in X_test_preprocessed])
    
    train_metrics_concat, test_metrics_concat = train_neural_network(
        X_train_glove_concat, y_train, X_test_glove_concat, y_test, input_dim=1000
    )
    
    print(f"Concatenated Feature Training Accuracy: {train_metrics_concat['accuracy']:.4f}")
    print(f"Concatenated Feature Training Precision: {train_metrics_concat['precision']:.4f}")
    print(f"Concatenated Feature Training Recall: {train_metrics_concat['recall']:.4f}")
    print(f"Concatenated Feature Training F1-score: {train_metrics_concat['f1']:.4f}")
    print(f"Concatenated Feature Testing Accuracy: {test_metrics_concat['accuracy']:.4f}")
    print(f"Concatenated Feature Testing Precision: {test_metrics_concat['precision']:.4f}")
    print(f"Concatenated Feature Testing Recall: {test_metrics_concat['recall']:.4f}")
    print(f"Concatenated Feature Testing F1-score: {test_metrics_concat['f1']:.4f}")

if __name__ == "__main__":
    main()