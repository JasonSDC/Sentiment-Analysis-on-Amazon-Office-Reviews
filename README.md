# Sentiment Analysis on Amazon Office Product Reviews

> Large-Scale Sentiment Analysis using Machine Learning and Deep Learning approaches on Amazon Office Product Reviews dataset (2.6M+ reviews)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-yellowgreen.svg)

## Project Overview

This project implements a comprehensive sentiment analysis pipeline that:
1. **Preprocesses** 2.6M+ Amazon reviews with text cleaning and NLP techniques
2. **Trains & evaluates** multiple ML classifiers (Perceptron, SVM, Logistic Regression)
3. **Explores** word embeddings using pre-trained GloVe vectors
4. **Compares** neural network architectures with different feature representations

## Dataset

- **Download**: [Google Drive](https://drive.google.com/drive/folders/1khdzXElRcIxVUR8s-4oQhDMzfeXInnl2?usp=sharing) (place the `.tsv.zip` file in the project root directory)
- **Source**: [Amazon Customer Reviews Dataset](https://www.amazon.com/review)
- **Product Category**: Office Products
- **Total Reviews**: 2,642,434
  - ✅ Positive (rating > 3): 2,002,886
  - ❌ Negative (rating < 3): 445,730
  - ⚪ Neutral (rating = 3): 193,818 *(excluded)*
- **Balanced Sample**: 200,000 reviews (100K positive + 100K negative)

## Technical Pipeline

### 1. Data Cleaning
- Convert text to lowercase
- Remove HTML tags and URLs
- Keep only alphabetic characters
- Remove extra whitespace

```
Average length before cleaning: 318.2539
Average length after cleaning:  300.4116
```

### 2. Text Preprocessing (NLTK)
- Stopword removal (English)
- Lemmatization using WordNetLemmatizer

```
Average length after preprocessing: 191.3101
```

### 3. Feature Extraction
- **TF-IDF Vectorization** with `min_df=2`, `max_df=0.95`
- Resulting feature matrix: `(160000, 34783)`

## Model Performance

### Traditional ML Models (TF-IDF Features)

| Model | Train Acc | Train Prec | Train Recall | Train F1 | Test Acc | Test Prec | Test Recall | Test F1 |
|-------|-----------|------------|--------------|----------|----------|-----------|-------------|---------|
| **Perceptron** | 0.9007 | 0.8937 | 0.9095 | 0.9015 | 0.8547 | 0.8464 | 0.8668 | 0.8565 |
| **SVM (LinearSVC)** | 0.9294 | 0.9313 | 0.9273 | 0.9293 | 0.8942 | 0.8954 | 0.8927 | 0.8940 |
| **Logistic Regression** | 0.9084 | 0.9119 | 0.9042 | 0.9080 | **0.8957** | 0.8988 | 0.8917 | **0.8953** |

> **Best Traditional Model**: Logistic Regression with **89.57% test accuracy**

### Neural Network with GloVe Embeddings

Using pre-trained `glove-wiki-gigaword-100` word vectors:

#### Semantic Similarity Examples
```
king - man + woman = queen, monarch, throne, daughter, princess
Outstanding = exceptional, achievement, award, best, contribution
```

#### Network Architecture
- Input → Linear(50) → ReLU → Dropout(0.2) → Linear(10) → ReLU → Dropout(0.2) → Linear(2)
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 20

| Feature Type | Train Acc | Train Prec | Train Recall | Train F1 | Test Acc | Test Prec | Test Recall | Test F1 |
|--------------|-----------|------------|--------------|----------|----------|-----------|-------------|---------|
| **Average Pooling (100-d)** | 0.8422 | 0.8590 | 0.8189 | 0.8385 | **0.8334** | 0.8507 | 0.8086 | 0.8291 |
| **Concatenated (1000-d)** | 0.8977 | 0.9006 | 0.8941 | 0.8974 | 0.7768 | 0.7792 | 0.7723 | 0.7758 |

> **Key Insight**: Average pooling (100-d) generalizes better than concatenation (1000-d) despite lower training accuracy, indicating the concatenation approach overfits to the training data.

## Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn nltk torch gensim
```

### Run the Analysis
```bash
python sentiment.py
```

> ⏱️ Note: First run will download NLTK data and GloVe embeddings (~400MB)

## Project Structure

```
Sentiment-Analysis-on-Amazon-Office-Reviews/
├── sentiment.py                                    # Main analysis script
├── amazon_reviews_us_Office_Products_v1_00.tsv.zip # Dataset (compressed)
└── README.md                                       # Project documentation
```

## Key Findings

1. **TF-IDF + Logistic Regression** achieves the best balance of accuracy (89.57%) and generalization
2. **Average GloVe pooling** provides robust representations that generalize well (83.34% test acc)
3. **Concatenated GloVe features** overfit significantly (89.77% train → 77.68% test)
4. Traditional ML models with TF-IDF outperform simple neural networks with word embeddings on this task

## Technologies Used

- **Data Processing**: Pandas, NumPy
- **NLP**: NLTK (stopwords, WordNetLemmatizer)
- **ML Models**: scikit-learn (Perceptron, LinearSVC, LogisticRegression, TfidfVectorizer)
- **Deep Learning**: PyTorch
- **Word Embeddings**: Gensim (GloVe)