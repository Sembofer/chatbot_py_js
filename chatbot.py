import random
import json
import pickle
import numpy as np
import time


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

def clean_up_sentence(sentence):
    nltk.data.path.append('./wordnet')
    nltk.data.path.append('./punkt_tab')
    nltk.data.path.append('./tokenizers')
    nltk.data.path.append('./corpora')
    nltk.data.path.append('./')
    #nltk.download('punkt_tab')
    #nltk.download('wordnet')
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.keras')
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



if __name__=="__main__":
    print("Chat listo!")
    while True:
        message = input("Usuario: ")
        if message == "salir":
            break
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)
