import os
import random

from process import tokenize


def load_data(
        directory: str = "aclImdb",
        mode: str = 'train',
) -> list:

    # Loading from file
    data_directory = f"{directory}/{mode}"
    labeled_data = []

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
                    data = f.read().replace("<br />", "")

                    # correcting misspelled words
                    # corrected_data = ''.join(correct_spelling(data))

                    # tokenization
                    filtered_data = tokenize(data)

                    # collecting tuples
                    if label == 'pos':
                        labeled_data.append((filtered_data, 1, rating_score))

                    else:
                        labeled_data.append((filtered_data, 0, rating_score))

    random.shuffle(labeled_data)

    return labeled_data

