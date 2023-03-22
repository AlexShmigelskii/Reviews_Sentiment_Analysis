import joblib
import warnings
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import settings
from process import tokenize, save_model, load_model
from loading_data import load_data
from visualizer import make_confusion_matrix, plot_roc_curves
from log import log_test

warnings.filterwarnings('ignore')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def train_model(
        necessary_accuracy: int = 0.8,
):
    # collecting data
    A = load_data(mode='train')

    cv = TfidfVectorizer(max_features=settings.max_features)

    # first model
    reg_sentiment_model = LogisticRegression()
    reg_rating_model = LogisticRegression()

    reviews = [i[0] for i in A]
    sentiment = [i[1] for i in A]
    rating = [i[2] for i in A]

    X = cv.fit_transform(reviews)

    x_train, x_test, y_train, y_test, r_train, r_test = train_test_split(X, sentiment, rating, test_size=0.001,
                                                                         random_state=42)

    # model fitting
    reg_sentiment_model.fit(x_train, y_train)
    reg_rating_model.fit(x_train, r_train)

    # testing the model
    reg_sentiment_pred = reg_sentiment_model.predict(x_test)
    reg_rating_pred = reg_rating_model.predict(x_test)

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

    save_model(reg_sentiment_model, filename=f'reg_sentiment_model_{settings.global_num}.joblib')
    save_model(reg_rating_model, filename=f'reg_rating_model_{settings.global_num}.joblib')
    save_model(cv, filename=f'vectorizer_{settings.global_num}.pkl')


def test_model():

    # loading models
    loaded_sentiment_model = load_model(filename=f'reg_sentiment_model_{settings.global_num}.joblib')
    loaded_rating_model = load_model(filename=f'reg_rating_model_{settings.global_num}.joblib')
    loaded_cv = load_model(filename=f'vectorizer_{settings.global_num}.pkl')

    B = load_data(mode='test')

    reviews_test = [i[0] for i in B]
    sentiment_test = [i[1] for i in B]
    rating_test = [i[2] for i in B]

    X = loaded_cv.transform(reviews_test)

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

    # writing results down to log.txt
    log_test(reg_sentiment_accuracy, reg_rating_accuracy,
             precision_sentiment, precision_rating,
             recall_sentiment, recall_rating,
             f1_sentiment, f1_rating
             )


def predict_single_review(
        review,
        sentiment_model=joblib.load('models/sentiment_models/reg_sentiment_model13.joblib'),
        rating_model=joblib.load('models/rating_models/reg_rating_model13.joblib'),
        cv=joblib.load('models/vect/vectorizer13.pkl'),
):
    """
    Определение только одного отзыва
    """

    filtered_review = tokenize(review)

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
