import os
import random
import warnings
import spacy
import nltk
import joblib

from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

warnings.filterwarnings('ignore')


# nltk.download('punkt')
# nltk.download('stopwords')


def load_training_data(
        data_directory: str = "aclImdb/train",
) -> list:
    # Loading from file
    stop_words = stopwords.words('english')
    labeled_data = []

    nlp = spacy.load("en_core_web_sm")
    for label in ["pos", "neg"]:

        labeled_directory = f"{data_directory}/{label}"

        for review in os.listdir(labeled_directory):

            if review.endswith(".txt"):

                # extract rating score
                rating_score = int(os.path.splitext(review)[0][-1])

                if rating_score == 0:
                    rating_score = int(os.path.splitext(review)[0][-2:])

                # extracting data
                with open(f"{labeled_directory}/{review}") as f:
                    data = f.read()
                    data = nlp(data.replace("<br />", ""))

                    # Tokenizaion
                    filtered_data = " ".join([str(token.lemma_).lower() for token in data if not token.is_stop
                                              and str(token) not in stop_words
                                              and str(token) not in ['.', ',', '-']])

                    if label == 'pos':
                        labeled_data.append((filtered_data, 1, rating_score))

                    else:
                        labeled_data.append((filtered_data, 0, rating_score))

    random.shuffle(labeled_data)

    return labeled_data


def load_test_data(
    data_directory: str = "aclImdb/test",
) -> list:
    # Loading from file
    stop_words = stopwords.words('english')
    labeled_data = []

    nlp = spacy.load("en_core_web_sm")
    for label in ["pos", "neg"]:

        labeled_directory = f"{data_directory}/{label}"

        count = 0

        for review in os.listdir(labeled_directory):

            count += 1
            if count > 20:
                break

            if review.endswith(".txt"):

                # extract rating score
                rating_score = int(os.path.splitext(review)[0][-1])

                if rating_score == 0:
                    rating_score = int(os.path.splitext(review)[0][-2:])

                # extracting data
                with open(f"{labeled_directory}/{review}") as f:
                    data = f.read()
                    data = nlp(data.replace("<br />", ""))

                    # Tokenizaion
                    filtered_data = " ".join([str(token.lemma_).lower() for token in data if not token.is_stop
                                              and str(token) not in stop_words
                                              and str(token) not in ['.', ',', '-']])

                    if label == 'pos':
                        labeled_data.append((filtered_data, 1, rating_score))

                    else:
                        labeled_data.append((filtered_data, 0, rating_score))

                    if count > 20:
                        break

    random.shuffle(labeled_data)

    return labeled_data


def make_word_cloud(
        data: list,
):
    consolidated = ' '.join(word for word in [tup[0] for tup in data if tup[1] == 0])

    word_cloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110)
    plt.figure(figsize=(15, 10))
    plt.imshow(word_cloud.generate(consolidated), interpolation='bilinear')
    plt.axis('off')
    plt.show()


# def make_confusion_matrix(
#
# ):
#     cm = confusion_matrix(y_test, reg_sentiment_pred)
#
#     cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
#
#     cm_display.plot()
#     plt.show()


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
    # collecting data
    A = load_training_data()

    reviews = [i[0] for i in A]
    sentiment = [i[1] for i in A]
    rating = [i[2] for i in A]

    cv = TfidfVectorizer()  # max_features=2500
    X = cv.fit_transform(reviews)

    # x_train, x_test, y_train, y_test, r_train, r_test = train_test_split(X, sentiment, rating, test_size=0.2,
    #                                                                      random_state=42)

    x_train = reviews
    y_train = sentiment
    r_train = rating

    # first model

    reg_sentiment_model = LogisticRegression()
    reg_rating_model = LogisticRegression()

    # model fitting
    reg_sentiment_model.fit(x_train, y_train)
    reg_rating_model.fit(x_train, r_train)

    # testing the model
    # reg_sentiment_pred = reg_sentiment_model.predict(x_test)
    # reg_rating_pred = reg_rating_model.predict(x_test)

    # model accuracy
    # reg_sentiment_accuracy = accuracy_score(y_test, reg_sentiment_pred)
    # reg_rating_accuracy = accuracy_score(r_test, reg_rating_pred)

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

    save_model(reg_sentiment_model, filename='reg_sentiment_model')
    save_model(reg_rating_model, filename='reg_rating_modelw')


def test_model():

    A = load_test_data()

    x_test = [i[0] for i in A]
    y_test = [i[1] for i in A]
    r_test = [i[2] for i in A]

    # loading models
    loaded_sentiment_model = load_model(filename='reg_sentiment_model_1_88.66%.sav')
    loaded_rating_model = load_model(filename='reg_rating_model_1_42.58%.sav')

    # testing the models
    reg_sentiment_pred = loaded_sentiment_model.predict(x_test)
    reg_rating_pred = loaded_rating_model.predict(x_test)

    # model accuracy
    reg_sentiment_accuracy = accuracy_score(y_test, reg_sentiment_pred)
    reg_rating_accuracy = accuracy_score(r_test, reg_rating_pred)

    print('LogisticRegression Sentiment accuracy - ', str(reg_sentiment_accuracy * 100) + '%')
    print('LogisticRegression Rating accuracy - ', str(reg_rating_accuracy * 100) + '%')


if __name__ == "__main__":

    test_model()
