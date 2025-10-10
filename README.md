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

Sentiment_Analysis_Project/
│
├── predict_news_interactive.py   # Script for interactive predictions
├── fake_news_model.h5            # Trained LSTM model
├── tokenizer.pkl                 # Tokenizer used for text processing
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── data/
    └── fake_news_dataset.csv     # Dataset used for training
