import settings


def log_test(sentiment, rating,
             precision_sentiment, precision_rating,
             recall_sentiment, recall_rating,
             f1_sentiment, f1_rating
             ):
    with open('log.txt', 'a') as f:
        f.write(f'Test "{settings.global_num}":\n')
        f.write(f'Max_features: {settings.max_features}\n')

        f.write(f'LogisticRegression Sentiment accuracy - {sentiment}\n')
        f.write(f'LogisticRegression Rating accuracy -  {rating}\n')

        f.write(f'LogisticRegression Sentiment precision_score - {precision_sentiment}\n')
        f.write(f'LogisticRegression Rating precision_score -  {precision_rating}\n')

        f.write(f'LogisticRegression Sentiment recall - {recall_sentiment}\n')
        f.write(f'LogisticRegression Rating recall -  {recall_rating}\n')

        f.write(f'LogisticRegression Sentiment f1-score - {f1_sentiment}\n')
        f.write(f'LogisticRegression Rating f1-score -  {f1_rating}\n\n')
