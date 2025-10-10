# Fake News Detection using Deep Learning (LSTM)

This project uses Natural Language Processing (NLP) and a Deep Learning LSTM model to classify news articles as **Real** or **Fake**.  
It demonstrates how to preprocess text, train a neural network, evaluate performance, and deploy an interactive prediction system.

---

## Project Overview

Fake news spreads quickly across digital platforms.  
The goal of this project is to build an AI model that can automatically detect whether a news article is fake or real based on its text content.

The model was trained using an LSTM (Long Short-Term Memory) neural network that learns patterns in text sequences.  
It was implemented in Python using TensorFlow and Keras.

---

## Key Features

- Text preprocessing with tokenization and padding
- LSTM-based neural network for classification
- Training with class balancing to handle uneven data
- Evaluation using confusion matrix, precision, recall, and F1-score
- Interactive prediction system for new unseen news articles

---

## Dataset

The dataset contains labeled news articles with two categories:
- `0`: Fake
- `1`: Real

Each record includes:
- **Title**
- **Text**
- **Label**

The title and text were combined to give more context before training.

---

## Project Structure
```bash
Sentiment_Analysis_Project/
│
├── fake_news_model.h5            # Trained LSTM model
├── tokenizer.pkl                 # Tokenizer used for text processing
├── README.md                     # Project documentation
└── data/
    └── fake_news_dataset.csv     # Dataset used for training
```

---

## Model Architecture

1. **Embedding Layer**: Converts words into dense vectors of fixed size.  
2. **LSTM Layer**: Captures sequential relationships in text.  
3. **Dense Layer (Sigmoid)**: Outputs probability between 0 and 1 for binary classification.

Code Samples Breakdown:
```Python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 10000
max_len = 200  # Increase to capture more context

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
```

