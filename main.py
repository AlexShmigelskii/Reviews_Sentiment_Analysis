import settings
from predict import train_model, test_model


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


if __name__ == "__main__":
    settings.init()  # создаем этот вызов один раз при создании программы

    train_model()
    test_model()
