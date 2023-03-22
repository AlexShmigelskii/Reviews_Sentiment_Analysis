import os
import sys

import joblib
import requests

from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def tokenize(
        data: str
) -> str:
    tokens = word_tokenize(data)
    return " ".join([lemmatizer.lemmatize(word.lower()) for word in tokens
                              if not word.lower() in stop_words
                              # and word.isalpha()
                              and word.isalnum()
                              ])


def correct_spelling(text):
    """
    Исправляет опечатки в переданной строке при помощи Yandex Speller API.
    """
    url = 'https://speller.yandex.net/services/spellservice.json/checkText'
    params = {
        'text': text,
        'options': 518,  # включает варианты исправлений слов с использованием словарей Яндекса
    }
    response = requests.get(url, params=params)
    data = response.json()
    fixed_text = text

    for error in reversed(data):
        start_pos = error['pos']
        end_pos = error['pos'] + error['len']
        fixed_text = fixed_text[:start_pos] + error['s'][0] + fixed_text[end_pos:]

    return fixed_text


def save_model(
        new_model,
        directory: str = 'model/',
        filename: str = "Completed_model.joblib"
):
    if 'sentiment' in filename:
        directory = 'models/sentiment_models/'

    elif 'rating' in filename:
        directory = 'models/rating_models/'

    elif 'vectorizer' in filename:
        directory = 'models/vect/'

    full_path = directory + filename
    joblib.dump(new_model, full_path)


def load_model(
        directory: str = 'models/',
        filename: str = "Completed_model.joblib"
):
    if 'sentiment' in filename:
        directory = 'models/sentiment_models/'

    elif 'rating' in filename:
        directory = 'models/rating_models/'

    elif 'vectorizer' in filename:
        directory = 'models/vect/'

    full_path = directory + filename

    if os.path.exists(full_path):
        loaded_model = joblib.load(full_path)
        return loaded_model

    else:
        print('FILE DOES NOT EXIST')
        sys.exit()
