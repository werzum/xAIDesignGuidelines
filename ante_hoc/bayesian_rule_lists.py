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
from sklearn.metrics import accuracy_score, confusion_matrix, mutual_info_score
from sklearn.model_selection import train_test_split

from ante_hoc.plots.bayesian_rule_list_plots import viz_classification_preds
from plots.confusion_matrix_plot import confusion_matrix_plot_sns

# [1.] Import data -----------------------------------------------------------------------------------------------------
data = loadarff(os.path.join(os.path.join("data", "input_data", "diabetes_dataset", "dataset_37_diabetes.arff")))
data_np = np.array(list(map(lambda x: np.array(list(x)), data[0])))
X = data_np[:, :-1].astype('float32')
y_text = data_np[:, -1].astype('str')
y = (y_text == 'tested_positive').astype(int)  # labels 0-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75)  # split
feature_names = ["#Pregnant", "Glucose concentration test", "Blood pressure(mmHg)",
                 "Triceps skin fold thickness(mm)",
                 "2-Hour serum insulin (mu U/ml)", "Body mass index", "Diabetes pedigree function", "Age (years)"]
features_nr = len(feature_names)
target_names = ['tested_negative', 'tested_positive']

features_selected_nr = 3
disc = ExtraBasicDiscretizer(feature_names[:features_selected_nr], n_bins=features_selected_nr, strategy='uniform')
X_train_brl_df = disc.fit_transform(pd.DataFrame(X_train[:, :features_selected_nr], columns=feature_names[:features_selected_nr]))
X_test_brl_df = disc.transform(pd.DataFrame(X_test[:, :features_selected_nr], columns=feature_names[:features_selected_nr]))
print("---------------------------------------------------------------------------------------------------------------")

# [2.] Bayesian Rule List Classifier -----------------------------------------------------------------------------------
bayesian_rule_list_classifier = BayesianRuleListClassifier(listlengthprior=2, listwidthprior=2, maxcardinality=2)
bayesian_rule_list_classifier.fit(X_train_brl_df.values, y_train, feature_names=X_train_brl_df.columns)

# [3.] Evaluation ------------------------------------------------------------------------------------------------------
probs = bayesian_rule_list_classifier.predict_proba(X_test_brl_df.values)
y_pred = bayesian_rule_list_classifier.predict(X_test_brl_df.values)

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

confusion_matrix_plot_sns(conf_matrix, "bayesian_rule_lists", features_nr, target_names)

# [4.] Print the learned rules -----------------------------------------------------------------------------------------
print(bayesian_rule_list_classifier)

# [5.] Visualization of test set performance ---------------------------------------------------------------------------
viz_classification_preds(probs, y_test)
