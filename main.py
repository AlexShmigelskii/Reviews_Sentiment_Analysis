import copy
import os
import random
import warnings
import time
import numpy as np
import spacy
import nltk
import joblib
import requests

from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, \
    roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

warnings.filterwarnings('ignore')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

global_num = 32
max_features = 20000
batch_size = 5000
num_epochs = 8


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


def load_training_data(
        data_directory: str = "aclImdb/train",
) -> list:
    # Loading from file
    labeled_data = []

    # nlp = spacy.load("en_core_web_sm")

    for label in ["pos", "neg"]:

        count = 0

        labeled_directory = f"{data_directory}/{label}"

        for review in os.listdir(labeled_directory):

            # count += 1
            # if count > 150:
            #     break

            if review.endswith(".txt"):

                # extract rating score
                rating_score = int(os.path.splitext(review)[0][-1])

                if rating_score == 0:
                    rating_score = int(os.path.splitext(review)[0][-2:])

                # extracting data
                with open(f"{labeled_directory}/{review}") as f:
                    data = f.read().replace("<br />", "")

                    # correcting misspelled words
                    # corrected_data = ''.join(correct_spelling(data))

                    # Tokenizaion
                    tokens = word_tokenize(data)
                    filtered_data = " ".join([lemmatizer.lemmatize(word.lower()) for word in tokens
                                              if not word.lower() in stop_words
                                              # and word.isalpha()
                                              and word.isalnum()
                                              ])

                    if label == 'pos':
                        labeled_data.append((filtered_data, 1, rating_score))

                    else:
                        labeled_data.append((filtered_data, 0, rating_score))

                    # if count > 150:
                    #     break

    random.shuffle(labeled_data)

    # labeled_data_2 = copy.deepcopy(labeled_data)
    #
    # random.shuffle(labeled_data_2)
    #
    # labeled_data += labeled_data_2

    return labeled_data


def load_test_data(
        data_directory: str = "aclImdb/test",
) -> list:
    # Loading from file
    labeled_data = []

    # nlp = spacy.load("en_core_web_sm")
    for label in ["pos", "neg"]:

        labeled_directory = f"{data_directory}/{label}"

        # count = 0

        for review in os.listdir(labeled_directory):

            # count += 1
            # if count > 500:
            #     break

            if review.endswith(".txt"):

                # extract rating score
                rating_score = int(os.path.splitext(review)[0][-1])

                if rating_score == 0:
                    rating_score = int(os.path.splitext(review)[0][-2:])

                # extracting data
                with open(f"{labeled_directory}/{review}") as f:
                    data = f.read().replace("<br />", "")
                    # data = nlp(data.replace("<br />", ""))

                    # Tokenizaion
                    tokens = word_tokenize(data)
                    filtered_data = " ".join([lemmatizer.lemmatize(word.lower()) for word in tokens
                                              if not word.lower() in stop_words
                                              # and word.isalpha()
                                              and word.isalnum()
                                              ])

                    # filtered_data = " ".join([str(token.lemma_).lower() for token in data if not token.is_stop
                    #                           and str(token) not in stop_words
                    #                           and str(token) not in ['.', ',', '-']])

                    if label == 'pos':
                        labeled_data.append((filtered_data, 1, rating_score))

                    else:
                        labeled_data.append((filtered_data, 0, rating_score))

                    # if count > 500:
                    #     break

    random.shuffle(labeled_data)

    return labeled_data


def get_batches(data):
    n = len(data)
    n_batches = n // batch_size
    batches = []

    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        batches.append(batch)

    if n % batch_size != 0:
        batch = data[n_batches * batch_size:]
        batches.append(batch)

    random.shuffle(batches)

    return batches


def make_word_cloud(
        data: list,
):
    consolidated = ' '.join(word for word in [tup[0] for tup in data if tup[1] == 0])

    word_cloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110)
    plt.figure(figsize=(15, 10))
    plt.imshow(word_cloud.generate(consolidated), interpolation='bilinear')
    plt.axis('off')
    plt.show()


def make_confusion_matrix(
        y_test,
        reg_sentiment_pred
):
    cm = confusion_matrix(y_test, reg_sentiment_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])

    cm_display.plot()
    # plt.show()
    plt.savefig(f'png/confusion_matrix_{global_num}.png')


# похоже, что это абсолютно бесполезная штука

# def plot_error_history(error_history):
#     """
#     Функция для построения графика количества ошибок от эпохи
#     error_history - список, содержащий значения ошибок для каждой эпохи обучения
#     """
#
#     sentiment = [i[0] for i in error_history]
#     rating = [i[1] for i in error_history]
#
#     plt.plot(sentiment)
#     plt.grid()
#     plt.title("График процента правильных предсказаний тональности")
#     plt.xlabel(f"Batch * {batch_size}")
#     plt.ylabel("Процент")
#     # plt.show()
#     plt.savefig(f'png/error_history_sentiment_{global_num}.png')
#
#     plt.clf()
#
#     plt.plot(rating)
#     plt.grid()
#     plt.title("График процента правильных предсказаний рейтинга")
#     plt.xlabel(f"Batch * {batch_size}")
#     plt.ylabel("Процент")
#     # plt.show()
#     plt.savefig(f'png/error_history_rating_{global_num}.png')


def plot_roc_curves(y_true_sentiment, y_pred_sentiment, y_true_rating, y_pred_rating):

    # Считаем площадь под ROC-кривой для каждой модели
    auc_sentiment = roc_auc_score(y_true_sentiment[25:], y_pred_sentiment)
    # auc_rating = roc_auc_score(y_true_rating[24:], y_pred_rating)

    # Строим ROC-кривые для каждой модели
    fpr_sentiment, tpr_sentiment, _ = roc_curve(y_true_sentiment[25:], y_pred_sentiment)
    # fpr_rating, tpr_rating, _ = roc_curve(y_true_rating[:len(y_pred_rating)], y_pred_rating)

    # Рисуем графики
    plt.clf()
    plt.plot(fpr_sentiment, tpr_sentiment, label=f'Sentiment ROC curve (AUC = {auc_sentiment:.2f})')
    # plt.plot(fpr_rating, tpr_rating, label=f'Rating ROC curve (AUC = {auc_rating:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.savefig(f'png/roc_curves_{global_num}.png')


def save_model(
        new_model,
        directory: str = 'model/',
        filename: str = "Completed_model.joblib"
):
    if 'sentiment' in filename:
        directory = 'models/sentiment_models/'

    elif 'rating' in filename:
        directory = 'models/rating_models/'

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

    full_path = directory + filename

    if os.path.exists(full_path):
        loaded_model = joblib.load(full_path)
        return loaded_model

    else:
        print('FILE DOES NOT EX IST')


def train_model(
        necessary_accuracy: int = 0.8,
):
    err = []

    # collecting data
    A = load_training_data()

    cv = TfidfVectorizer(max_features=max_features)  # max_features=2500

    # first model
    reg_sentiment_model = LogisticRegression()
    reg_rating_model = LogisticRegression()

    for epoch in range(num_epochs):

        batches = get_batches(A)
        random.shuffle(batches)

        for batch in batches:
            reviews = [i[0] for i in batch]
            sentiment = [i[1] for i in batch]
            rating = [i[2] for i in batch]

            X = cv.fit_transform(reviews)

            x_train, x_test, y_train, y_test, r_train, r_test = train_test_split(X, sentiment, rating, test_size=0.3,
                                                                                 random_state=42)

            # model fitting
            reg_sentiment_model.fit(x_train, y_train)
            reg_rating_model.fit(x_train, r_train)

            # testing the model
            reg_sentiment_pred = reg_sentiment_model.predict(x_test)
            reg_rating_pred = reg_rating_model.predict(x_test)

            # model accuracy
            reg_sentiment_accuracy = accuracy_score(y_test, reg_sentiment_pred)
            reg_rating_accuracy = accuracy_score(r_test, reg_rating_pred)

            err.append((reg_sentiment_accuracy, reg_rating_accuracy))

    # print('LogisticRegression Sentiment accuracy - ', str(reg_sentiment_accuracy * 100) + '%')
    # print('LogisticRegression Rating accuracy - ', str(reg_rating_accuracy * 100) + '%')

    # second model

    # svm_model = LinearSVC()
    # svm_model.fit(x_train, y_train)
    # svm_pred = svm_model.predict(x_test)
    # svm_accuracy = accuracy_score(y_test, svm_pred)
    # print('SVM accuracy - ', str(svm_accuracy * 100) + '%')

    # Saving model
    # if reg_sentiment_accuracy > necessary_accuracy:
    #     save_model(reg_sentiment_model, filename='reg_sentiment_model_2' + str(reg_sentiment_accuracy * 100) + '%')
    #     save_model(reg_rating_model, filename='reg_rating_model_2' + str(reg_rating_accuracy * 100) + '%')

    save_model(reg_sentiment_model, filename=f'reg_sentiment_model_{global_num}.joblib')
    save_model(reg_rating_model, filename=f'reg_rating_model_{global_num}.joblib')

    return cv, err


def test_model(cv):

    # loading models
    loaded_sentiment_model = load_model(filename=f'reg_sentiment_model_{global_num}.joblib')
    loaded_rating_model = load_model(filename=f'reg_rating_model_{global_num}.joblib')

    B = load_test_data()

    reviews_test = [i[0] for i in B]
    sentiment_test = [i[1] for i in B]
    rating_test = [i[2] for i in B]

    X = cv.transform(reviews_test)

    _, x_test, _, y_test, _, r_test = train_test_split(X, sentiment_test, rating_test, test_size=0.999, shuffle=False)

    # testing the models
    reg_sentiment_pred = loaded_sentiment_model.predict(x_test)
    reg_rating_pred = loaded_rating_model.predict(x_test)

    # model accuracy
    reg_sentiment_accuracy = accuracy_score(y_test, reg_sentiment_pred)
    precision_sentiment = precision_score(y_test, reg_sentiment_pred, average='weighted')
    recall_sentiment = recall_score(y_test, reg_sentiment_pred, average='weighted')
    f1_sentiment = f1_score(y_test, reg_sentiment_pred, average='weighted')

    reg_rating_accuracy = accuracy_score(r_test, reg_rating_pred)
    precision_rating = precision_score(r_test, reg_rating_pred, average='weighted')
    recall_rating = recall_score(r_test, reg_rating_pred, average='weighted')
    f1_rating = f1_score(r_test, reg_rating_pred, average='weighted')

    # visualizing results
    make_confusion_matrix(y_test=y_test, reg_sentiment_pred=reg_sentiment_pred)
    plot_roc_curves(y_true_sentiment=sentiment_test, y_pred_sentiment=reg_sentiment_pred,
                    y_true_rating=rating_test, y_pred_rating=reg_rating_pred)

    print('LogisticRegression Sentiment accuracy - ', str(reg_sentiment_accuracy * 100) + '%')
    print('LogisticRegression Rating accuracy - ', str(reg_rating_accuracy * 100) + '%\n')

    print('LogisticRegression Sentiment precision_score - ', str(precision_sentiment * 100) + '%')
    print('LogisticRegression Rating precision_score - ', str(precision_rating * 100) + '%\n')

    print('LogisticRegression Sentiment recall - ', str(recall_sentiment * 100) + '%')
    print('LogisticRegression Rating recall - ', str(recall_rating * 100) + '%\n')

    print('LogisticRegression Sentiment f1-score - ', str(f1_sentiment * 100) + '%')
    print('LogisticRegression Rating f1-score - ', str(f1_rating * 100) + '%')

    log_test(reg_sentiment_accuracy, reg_rating_accuracy,
             precision_sentiment, precision_rating,
             recall_sentiment, recall_rating,
             f1_sentiment, f1_rating
             )


def predict_review(
        review,
        sentiment_model=joblib.load('models/sentiment_models/reg_sentiment_model13.joblib'),
        rating_model=joblib.load('models/rating_models/reg_rating_model13.joblib'),
        cv=joblib.load('models/vect/vectorizer13.pkl'),

):
    tokens = word_tokenize(review)
    filtered_review = " ".join([lemmatizer.lemmatize(word.lower()) for word in tokens
                                if not word.lower() in stop_words
                                and word.isalpha()])

    X = cv.transform([filtered_review])
    y_new = sentiment_model.predict(X)
    r_new = rating_model.predict(X)

    sentiment = 'negative ' if y_new == 0 else 'positive'
    rating = int(r_new)

    if (y_new == 0 and rating > 4) or (y_new == 1 and rating < 5):
        print(f"sorry, but I'm not sure about the decision... I guess that this review is {sentiment} but I gave it the"
              f"mark of {rating}")

    else:
        print(f"I guess that this review is {sentiment}. I gave it the mark of {rating}")


def log_test(sentiment, rating,
             precision_sentiment, precision_rating,
             recall_sentiment, recall_rating,
             f1_sentiment, f1_rating
             ):
    with open('log.txt', 'a') as f:
        f.write(f'Test "{global_num}":\n')
        f.write(f'Batch size: {batch_size};  Number of Epochs: {num_epochs};  Max_features: {max_features}\n')

        f.write(f'LogisticRegression Sentiment accuracy - {sentiment}\n')
        f.write(f'LogisticRegression Rating accuracy -  {rating}\n')

        f.write(f'LogisticRegression Sentiment precision_score - {precision_sentiment}\n')
        f.write(f'LogisticRegression Rating precision_score -  {precision_rating}\n')

        f.write(f'LogisticRegression Sentiment recall - {recall_sentiment}\n')
        f.write(f'LogisticRegression Rating recall -  {recall_rating}\n')

        f.write(f'LogisticRegression Sentiment f1-score - {f1_sentiment}\n')
        f.write(f'LogisticRegression Rating f1-score -  {f1_rating}\n\n')


if __name__ == "__main__":
    # vectorizer, errors = train_model()
    # joblib.dump(vectorizer, f'models/vect/vectorizer_{global_num}.pkl')

    # plot_error_history(errors)

    vectorizer = joblib.load(f'models/vect/vectorizer_{global_num}.pkl')
    test_model(cv=vectorizer)

    # predict_review(review='''this was was awful, i freaking hate it''')
