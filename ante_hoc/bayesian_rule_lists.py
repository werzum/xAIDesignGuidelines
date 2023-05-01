"""
    Bayesian Rule Lists (BRL)

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-04-30
"""

import os

from imodels import BayesianRuleListClassifier
from imodels.discretization import ExtraBasicDiscretizer
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ante_hoc.plots.ante_hoc_plots import viz_classification_preds

# [1.] Import data -----------------------------------------------------------------------------------------------------
iris_data = load_iris()

feature_names = iris_data.feature_names
features_nr = len(feature_names)

target_names = iris_data.target_names
print(f"Feature names: {feature_names}, nr. of features: {features_nr}")
print(f"Target names: {target_names}")

X = iris_data.data
y = iris_data.target

feature_names = iris_data.feature_names
features_nr = len(feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

disc = ExtraBasicDiscretizer(feature_names[:3], n_bins=3, strategy='uniform')
X_train_brl_df = disc.fit_transform(pd.DataFrame(X_train[:, :3], columns=feature_names[:3]))
X_test_brl_df = disc.transform(pd.DataFrame(X_test[:, :3], columns=feature_names[:3]))

# [2.] Bayesian Rule List Classifier -----------------------------------------------------------------------------------
bayesian_rule_list_classifier = BayesianRuleListClassifier()
bayesian_rule_list_classifier.fit(X_train_brl_df.values, y_train,
                                  feature_names=X_train_brl_df.columns)

# [3.] Evaluation ------------------------------------------------------------------------------------------------------
probs = bayesian_rule_list_classifier.predict_proba(X_test_brl_df.values)
print(probs)

# [4.] Print the learned rules -----------------------------------------------------------------------------------------
print(bayesian_rule_list_classifier)

# [5.] Visualization of test set performance ---------------------------------------------------------------------------
viz_classification_preds(probs, y_test)
