import json
import random
import numpy as np
import nltk
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Read intents and responses from the JSON file
with open('intents.json') as file:
    data = json.load(file)

training_data = []
responses = {}
for ech in data['intents']:
    for ptn in ech['patterns']:
        training_data.append((ptn, ech['tag']))
    
    responses[ech['tag']] = ech['responses']

# Preprocess the training data
corpus = [data[0] for data in training_data]
intents = [data[1] for data in training_data]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

# Pad the sequences to make them of equal length
max_sequence_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert intents to numerical labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(intents)

# LSTM model for intent recognition
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(len(set(Y)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=50, batch_size=1)

# Define a threshold for intent prediction probability (adjust as needed)
intent_threshold = 0.7

def lstmAnswer(user_input=''):
    # Preprocess the user input
    input_sequence = tokenizer.texts_to_sequences([user_input])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)

    # Predict the intent using the trained model
    predicted_probs = model.predict(input_sequence_padded)
    predicted_label = np.argmax(predicted_probs, axis=1)[0]
    predicted_intent = label_encoder.inverse_transform([predicted_label])[0]

    # Get the probability of the predicted intent
    predicted_intent_prob = predicted_probs[0][predicted_label]

    # Check if the predicted intent has a high enough probability
    if predicted_intent_prob >= intent_threshold:
        # Generate and display the response based on the predicted intent
        response_array = responses.get(predicted_intent)
        if response_array:
            response = random.choice(response_array)
        else:
            response = "I'm sorry, I don't have the answer to that question."
    else:
        response = "I'm sorry, I don't have the answer to that question."

    return response    