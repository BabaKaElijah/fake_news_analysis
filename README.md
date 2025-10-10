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
├── Analysis_Code                 # File Containing all analysis code
├── Input_Code                    # Input file to run new text
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

## Code Samples Breakdown:
### Text Preprocessing
Before feeding text data into the LSTM model, all news articles are converted into numerical form using a Tokenizer and padding. Neural networks work with numbers, not raw text, so this step transforms text into sequences the model can understand.
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
### Building the LSTM Model
The LSTM model is designed to classify news articles as real or fake. It processes sequences of words and learns patterns in the text.
```Python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
### Handling Class Imbalance
Some datasets have more examples of one class than the other. This can make the model biased toward the majority class. To fix this, we compute class weights so the model gives more importance to the minority class.
```Python
from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)
```
### Training the Model
The model is trained on the preprocessed news articles using the LSTM network.
```Python
history = model.fit(
    X_train_pad, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    class_weight=class_weights_dict,
    verbose=1
)
```
### Evaluating the Model
After training, the model’s performance is tested on unseen data.
```Python
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Loss: {test_loss}")
```
### Predicting New Articles
This section demonstrates how to use the trained LSTM model to predict whether new news articles are real or fake.
```Python
# Example new articles
new_articles = [
    "The government announced a new education policy starting next month.",
    "Local team wins the national soccer championship.",
    "Miracle cure for diabetes discovered in remote village.",
    "Stock market sees steady growth after quarterly earnings report."
]

# Clean, tokenize, and pad
new_clean = [clean_text(text) for text in new_articles]
new_seq = tokenizer.texts_to_sequences(new_clean)
new_pad = pad_sequences(new_seq, maxlen=max_len)

# Predict
predictions = (model.predict(new_pad) > 0.5).astype("int32")

# Show results
for text, pred in zip(new_articles, predictions):
    label = "Fake" if pred[0] == 1 else "Real"
    print(f"Article: {text}\nPrediction: {label}\n")
```
### Interactive News Prediction
This script allows users to input news articles interactively and get predictions in real-time.
```Python
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load the trained model
model = load_model("fake_news_model.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Set the same max length used during training
MAX_LEN = 100

# Optional: same cleaning function used during training
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Function to predict news authenticity
def predict_news(article):
    article_clean = clean_text(article)
    seq = tokenizer.texts_to_sequences([article_clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = model.predict(padded)[0][0]
    label = "Real" if prob > 0.5 else "Fake"
    print(f"\nArticle: {article}")
    print(f"Prediction: {label}, Probability: {prob:.4f}")

# Main program
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a news article (or type 'exit' to quit):\n")
        if user_input.lower() == "exit":
            print("Exiting program.")
            break
        predict_news(user_input)
```
