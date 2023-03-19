import copy
import os
import random
import warnings
import spacy
import nltk
import joblib

from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud

warnings.filterwarnings('ignore')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def load_training_data(
        directory: str = "aclImdb/",
) -> list:
    # Loading from file
    labeled_data = []

    # nlp = spacy.load("en_core_web_sm")
    for mode in ['train', 'test']:

        data_directory = f"{directory}/{mode}"

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

                        # Tokenizaion
                        tokens = word_tokenize(data)
                        filtered_data = " ".join([lemmatizer.lemmatize(word.lower()) for word in tokens
                                                  if not word.lower() in stop_words
                                                  and word.isalpha()])

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
                                              and word.isalpha()])

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

    cv = TfidfVectorizer(max_features=25000)  # max_features=2500
    X = cv.fit_transform(reviews)

    x_train, x_test, y_train, y_test, r_train, r_test = train_test_split(X, sentiment, rating, test_size=0.0001,
                                                                         random_state=42)

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

    save_model(reg_sentiment_model, filename='reg_sentiment_model14.joblib')
    save_model(reg_rating_model, filename='reg_rating_model14.joblib')

    return cv


def test_model(cv):
    B = load_test_data()

    reviews_test = [i[0] for i in B]
    sentiment_test = [i[1] for i in B]
    rating_test = [i[2] for i in B]

    X = cv.transform(reviews_test)

    x_train, x_test, y_train, y_test, r_train, r_test = train_test_split(X, sentiment_test, rating_test, test_size=0.999,
                                                                         random_state=42)

    # loading models
    loaded_sentiment_model = load_model(filename='reg_sentiment_model14.joblib')
    loaded_rating_model = load_model(filename='reg_rating_model14.joblib')

    # testing the models
    reg_sentiment_pred = loaded_sentiment_model.predict(x_test)
    reg_rating_pred = loaded_rating_model.predict(x_test)

    # model accuracy
    reg_sentiment_accuracy = accuracy_score(y_test, reg_sentiment_pred)
    reg_rating_accuracy = accuracy_score(r_test, reg_rating_pred)

    print('LogisticRegression Sentiment accuracy - ', str(reg_sentiment_accuracy * 100) + '%')
    print('LogisticRegression Rating accuracy - ', str(reg_rating_accuracy * 100) + '%')


def predict_review(
        review,
        sentiment_model=joblib.load('models/sentiment_models/reg_sentiment_model13.joblib'),
        rating_model=joblib.load('models/rating_models/reg_rating_model13.joblib'),
        cv=joblib.load('vectorizer13.pkl'),

):
    tokens = word_tokenize(review)
    filtered_review = " ".join([lemmatizer.lemmatize(word.lower()) for word in tokens
                                if not word.lower() in stop_words
                                and word.isalpha()])

    X = cv.transform([filtered_review])
    y_new = sentiment_model.predict(X)
    r_new = rating_model.predict(X)

    sentiment = 'negative 'if y_new == 0 else 'positive'
    rating = int(r_new)

    if (y_new == 0 and rating > 4) or (y_new == 1 and rating < 5):
        print(f"sorry, but I'm not sure about the decision... I guess that this review is {sentiment} but I gave it the"
              f"mark of {rating}")

    else:
        print(f"I guess that this review is {sentiment}. I gave it the mark of {rating}")


if __name__ == "__main__":
    vectorizer = train_model()
    joblib.dump(vectorizer, 'vectorizer14.pkl')

    vectorizer = joblib.load('vectorizer14.pkl')
    test_model(cv=vectorizer)

    # predict_review(review='''''')


