import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# Read intents and responses from the JSON file
with open('intents.json') as file:
    data = json.load(file)

training_data = []
responses = {}
for i, ech in enumerate(data['intents']):

    for j, ptn in enumerate(ech['patterns']):
        tpl = ()
        tpl = tpl + (ptn, ech['tag'],)
        training_data.append(tpl)

    rpd = []
    for k, res in enumerate(ech['responses']):
        rpd.append(res)
    responses[ech['tag']] = rpd

# Preprocess the training data
corpus = [data[0] for data in training_data]
intents = [data[1] for data in training_data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Train the intent recognition model
naive_bayes = MultinomialNB()
naive_bayes.fit(X, intents)

def get_intent(text):
    # Preprocess the user input
    input_vector = vectorizer.transform([text])

    # Predict the intent using the trained model
    predicted_intent_probs = naive_bayes.predict_proba(input_vector)
    max_prob_index = np.argmax(predicted_intent_probs)
    max_prob = predicted_intent_probs[0][max_prob_index]

    # Set a threshold for intent prediction probability (adjust as needed)
    intent_threshold = 0.6

    if max_prob >= intent_threshold:
        predicted_intent = naive_bayes.classes_[max_prob_index]
        return predicted_intent
    else:
        return None

def replace_words_with_space(input_string):
    words_to_replace = {'is', 'for', 'some', 'to', 'please', 'the', 'are', 'can', 'get'}
    words_to_replace = {' ' + word + ' ' for word in words_to_replace}

    # Split the input string into words
    words = input_string.split()

    # Create a set of words to replace for faster lookups
    words_to_replace_set = set(words_to_replace)

    # Replace the specified words with a space
    replaced_words = [' ' if word in words_to_replace_set else word for word in words]

    # Join the words back into a string
    result_string = ' '.join(replaced_words)

    return result_string


def get_cosine_similarity(text):
    input_vector = vectorizer.transform([text])
    similarities = cosine_similarity(input_vector, X)
    max_similarity_index = np.argmax(similarities)
    max_similarity = similarities[0][max_similarity_index]

    # Set a threshold for cosine similarity (adjust as needed)
    similarity_threshold = 0.5

    if max_similarity >= similarity_threshold:
        predicted_intent = intents[max_similarity_index]
        return predicted_intent
    else:
        return None

def nbAnswer(user_input=''):
    user_input = replace_words_with_space(user_input)

    # Get the most probable intent
    predicted_intent = get_intent(user_input)

    if not predicted_intent:
        # Fallback using cosine similarity
        predicted_intent = get_cosine_similarity(user_input)

    if predicted_intent:
        response_array = responses.get(predicted_intent)
        if response_array:
            response = random.choice(response_array)
        else:
            response = "I'm sorry, I don't have the answer to that question."        
    else:
        response = "I'm sorry, I don't have the answer to that question."

    return response


