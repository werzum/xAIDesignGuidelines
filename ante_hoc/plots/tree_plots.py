"""
    Plots for tree classifiers/regressors

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-04-31
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree


def plot_feature_importances_hist_matplotlib(features_column_names: list,
                                             feature_importances: list):
    """
    Plot the feature importances in a horizontal histogram with matplotlib

    :param features_column_names: Features column names
    :param feature_importances: The corresponding feature importances
    :return:
    """

    output_data_path = os.path.join(os.path.join("data", "output_data", "plots"))

    fig, ax = plt.subplots(figsize=(12, 12))

    y_pos = np.arange(len(features_column_names))

    ax.barh(y_pos, feature_importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_column_names, fontsize=24, fontweight='bold')
    ax.invert_yaxis()
    ax.set_title('Feature Importances', fontsize=32, fontweight='bold')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_data_path, f"explanations_feature_importances.png"))
    plt.close()


def plot_decision_tree_rules(decision_tree_classifier: tree.DecisionTreeClassifier,
                             features_column_names: list,
                             task: str,
                             features_nr: int):
    """
    Visualize the hierarchy of features

    :param decision_tree_classifier: Decision tree regressor
    :param features_column_names: Features column names
    :param task: classification or regression
    :param features_nr: Number of features used for the prediction
    :return:
    """

    output_data_path = os.path.join(os.path.join("data", "output_data", "plots"))

    fig = plt.figure(figsize=(30, 25))
    tree.plot_tree(
        decision_tree_classifier,
        # max_depth=2,                          # MAX_DEPTH can be set -------------------------------------------------
        feature_names=features_column_names,
        filled=True,
        rounded=True,
        fontsize=30
    )
    plt.show()
    fig.savefig(os.path.join(output_data_path, f"Explanations DT tree with {features_nr} features.png"))
    plt.close()

