"""
    Decision Tree classification

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-04-31
"""

import os

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, mutual_info_score
from sklearn.model_selection import GridSearchCV, train_test_split

from plots.confusion_matrix_plot import confusion_matrix_plot_sns
from plots.tree_plots import plot_feature_importances_hist_matplotlib, plot_decision_tree_rules


# [1.] Import data -----------------------------------------------------------------------------------------------------
iris_data = load_iris()

feature_names = iris_data.feature_names
features_nr = len(feature_names)

target_names = iris_data.target_names
print(f"Feature names: {feature_names}, nr. of features: {features_nr}")
print(f"Target names: {target_names}")

X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# [2.] Decision Tree Classifier ----------------------------------------------------------------------------------------
dt_search_classifier = tree.DecisionTreeClassifier()
param_grid = [{
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "max_features": ["auto", "sqrt", "log2"],
    "criterion": ["gini", "entropy"]
}]

print("Start the Grid Search for the parameters")
grid_search = GridSearchCV(dt_search_classifier, param_grid, cv=4)
grid_search.fit(X_train, y_train)

print(f"Best parameters for the decision tree: {grid_search.best_params_}")
dt_classifier = grid_search.best_estimator_

y_pred = dt_classifier.predict(X_test)

# [3.] Classification metrics ------------------------------------------------------------------------------------------
# [3.1.] Accuracy ------------------------------------------------------------------------------------------------------
acc_score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc_score}")

# [3.2.] Mutual Information (MI) ---------------------------------------------------------------------------------------
mi = mutual_info_score(y_test, y_pred.reshape(1, -1)[0])
print(f"Mutual Info: {mi}")

# [3.3.] Confusion Matrix - Create an annotated version ----------------------------------------------------------------
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

confusion_matrix_plot_sns(conf_matrix, "decision tree", features_nr, target_names)

# [4.] Feature importance ----------------------------------------------------------------------------------------------
feature_importances = dt_classifier.feature_importances_

feature_importances_reverse_idxs = list(reversed(np.argsort(feature_importances)))
feature_names_sorted_importance = [feature_names[idx] for idx in feature_importances_reverse_idxs]

feature_importances_reverse_sort = sorted(feature_importances, reverse=True)

print(feature_names_sorted_importance)
print(feature_importances_reverse_sort)

plot_feature_importances_hist_matplotlib(feature_names_sorted_importance, feature_importances_reverse_sort)

# [5.] The computed decision tree --------------------------------------------------------------------------------------
plot_decision_tree_rules(dt_classifier, feature_names, "classification", features_nr)

# [6.] Text representation of the decision tree rules ------------------------------------------------------------------
text_representation = tree.export_text(dt_classifier, feature_names=feature_names)
print(text_representation)
