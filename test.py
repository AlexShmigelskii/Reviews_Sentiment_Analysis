# import os
# import random
# import warnings
# import spacy
# import nltk
# import joblib
#
# from matplotlib import pyplot as plt
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# from wordcloud import WordCloud
#
#
# def load_training_data(
#         data_directory: str = "aclImdb/train",
# ) -> list:
#     # Loading from file
#     stop_words = stopwords.words('english')
#     labeled_data = []
#
#     nlp = spacy.load("en_core_web_sm")
#     for label in ["pos", "neg"]:
#
#         count = 0
#
#         labeled_directory = f"{data_directory}/{label}"
#
#         for review in os.listdir(labeled_directory):
#
#             count += 1
#             if count > 500:
#                 break
#
#             if review.endswith(".txt"):
#
#                 # extract rating score
#                 rating_score = int(os.path.splitext(review)[0][-1])
#
#                 if rating_score == 0:
#                     rating_score = int(os.path.splitext(review)[0][-2:])
#
#                 # extracting data
#                 with open(f"{labeled_directory}/{review}") as f:
#                     data = f.read()
#                     data = nlp(data.replace("<br />", ""))
#
#                     # Tokenizaion
#                     filtered_data = " ".join([str(token.lemma_).lower() for token in data if not token.is_stop
#                                               and str(token) not in stop_words
#                                               and str(token) not in ['.', ',', '-']])
#
#                     if label == 'pos':
#                         labeled_data.append((filtered_data, 1, rating_score))
#
#                     else:
#                         labeled_data.append((filtered_data, 0, rating_score))
#
#                     if count > 500:
#                         break
#
#     random.shuffle(labeled_data)
#
#     return labeled_data
#
#
# def load_test_data(
#     data_directory: str = "aclImdb/test",
# ) -> list:
#     # Loading from file
#     stop_words = stopwords.words('english')
#     labeled_data = []
#
#     nlp = spacy.load("en_core_web_sm")
#     for label in ["pos", "neg"]:
#
#         labeled_directory = f"{data_directory}/{label}"
#
#         count = 0
#
#         for review in os.listdir(labeled_directory):
#
#             count += 1
#             if count > 100:
#                 break
#
#             if review.endswith(".txt"):
#
#                 # extract rating score
#                 rating_score = int(os.path.splitext(review)[0][-1])
#
#                 if rating_score == 0:
#                     rating_score = int(os.path.splitext(review)[0][-2:])
#
#                 # extracting data
#                 with open(f"{labeled_directory}/{review}") as f:
#                     data = f.read()
#                     data = nlp(data.replace("<br />", ""))
#
#                     # Tokenizaion
#                     filtered_data = " ".join([str(token.lemma_).lower() for token in data if not token.is_stop
#                                               and str(token) not in stop_words
#                                               and str(token) not in ['.', ',', '-']])
#
#                     if label == 'pos':
#                         labeled_data.append((filtered_data, 1, rating_score))
#
#                     else:
#                         labeled_data.append((filtered_data, 0, rating_score))
#
#                     if count > 100:
#                         break
#
#     random.shuffle(labeled_data)
#
#     return labeled_data
#
#
# def train_model(
#         necessary_accuracy: int = 0.8,
# ):
#     # collecting data
#     A = load_training_data()
#
#     reviews_train = [i[0] for i in A]
#     sentiment_train = [i[1] for i in A]
#     rating_train = [i[2] for i in A]
#
#     cv = TfidfVectorizer(max_features=5000)
#     X = cv.fit_transform(reviews_train)
#
#     x_train, x_test, y_train, y_test, r_train, r_test = train_test_split(X, sentiment_train, rating_train, test_size=0.1,
#                                                                          random_state=42)
#
#     # x_train = X
#     # y_train = sentiment
#     # r_train = rating
#
#     # first model
#
#     reg_sentiment_model = LogisticRegression()
#     reg_rating_model = LogisticRegression()
#
#     # model fitting
#     reg_sentiment_model.fit(x_train, y_train)
#     reg_rating_model.fit(x_train, r_train)
#
#     B = load_test_data()
#
#     reviews_test = [i[0] for i in B]
#     sentiment_test = [i[1] for i in B]
#     rating_test = [i[2] for i in B]
#
#     XX = cv.transform(reviews_test)
#
#     x_train, x_test, y_train, y_test, r_train, r_test = train_test_split(XX, sentiment_test, rating_test,
#                                                                          test_size=1,
#                                                                          random_state=42)
#
#     reg_sentiment_pred = reg_sentiment_model.predict(x_test)
#     reg_rating_pred = reg_rating_model.predict(x_test)
#
#     # model accuracy
#     reg_sentiment_accuracy = accuracy_score(y_test, reg_sentiment_pred)
#     reg_rating_accuracy = accuracy_score(r_test, reg_rating_pred)
#
#     print('LogisticRegression Sentiment accuracy - ', str(reg_sentiment_accuracy * 100) + '%')
#     print('LogisticRegression Rating accuracy - ', str(reg_rating_accuracy * 100) + '%')
#
#
# train_model()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curves(y_true, y_pred_sentiment, y_pred_rating):
    # Считаем площадь под ROC-кривой для каждой модели
    auc_sentiment = roc_auc_score(y_true, y_pred_sentiment)
    auc_rating = roc_auc_score(y_true, y_pred_rating)

    # Строим ROC-кривые для каждой модели
    fpr_sentiment, tpr_sentiment, _ = roc_curve(y_true, y_pred_sentiment)
    fpr_rating, tpr_rating, _ = roc_curve(y_true, y_pred_rating)

    # Рисуем графики
    plt.plot(fpr_sentiment, tpr_sentiment, label=f'Sentiment ROC curve (AUC = {auc_sentiment:.2f})')
    plt.plot(fpr_rating, tpr_rating, label=f'Rating ROC curve (AUC = {auc_rating:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.show()


# y_true - истинные метки классов
# y_pred_sentiment - предсказанные вероятности классов для определения тональности
# y_pred_rating - предсказанные вероятности классов для определения рейтинга
y_true = np.array([1, 0, 1, 0, 1, 0])
y_pred_sentiment = np.array([0.8, 0.2, 0.9, 0.3, 0.7, 0.4])
y_pred_rating = np.array([0.6, 0.4, 0.8, 0.2, 0.9, 0.1])

plot_roc_curves(y_true, y_pred_sentiment, y_pred_rating)
