import settings
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


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
    plt.savefig(f'png/confusion_matrix_{settings.global_num}.png')


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
    plt.savefig(f'png/roc_curves_{settings.global_num}.png')


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

