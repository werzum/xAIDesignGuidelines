"""
    Useful plots for ante-hoc methods

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-04-30
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def viz_classification_preds(probs, y_test):
    """
    Visualization for classification prediction.
    Most of it overtaken by imodels.

    :param probs:
    :param y_test:
    :return:
    """

    plt.subplot(121)
    plt.hist(probs[:, 1][y_test == 0], label='Class 0')
    plt.hist(probs[:, 1][y_test == 1], label='Class 1', alpha=0.8)
    plt.ylabel('Count')
    plt.xlabel('Predicted probability of class 1')
    plt.legend()

    plt.subplot(122)
    preds = np.argmax(probs, axis=1)
    plt.title('ROC curve')
    fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot(fpr, tpr)
    plt.tight_layout()
    plt.show()
    plt.close()
