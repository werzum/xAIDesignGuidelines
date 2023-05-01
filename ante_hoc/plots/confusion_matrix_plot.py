"""
    Plot confusion matrix

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-04-30
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def confusion_matrix_plot_sns(conf_matrix: np.array, method: str, features_nr: int, target_names: list):
    """
    Plot the confusion matrix

    :param conf_matrix: Input confusion matrix
    :param method: Method that was used in classification
    :param features_nr: Number of features
    :param target_names: List of target names
    :param method: Method used (decision tree, gam, bayesian rule lists)
    :return:
    """

    output_data_path = os.path.join(os.path.join("data", "output_data", "plots", method))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', annot_kws={"size": 32})

    ax.set_title(f'Confusion matrix with {method} \n\n', fontsize=32, fontweight='bold')
    ax.set_xlabel('\nPredicted Values', fontsize=28, fontweight='bold')
    ax.set_ylabel('Actual Values ', fontsize=28, fontweight='bold')

    ax.xaxis.set_ticklabels(target_names, fontsize=28, fontweight='bold')
    ax.yaxis.set_ticklabels(target_names, fontsize=28, fontweight='bold')

    plt.show()
    fig.savefig(os.path.join(output_data_path, f"confusion_matrix_{method}.png"))
    plt.close()
