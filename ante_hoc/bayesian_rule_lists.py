"""
    Bayesian Rule Lists (BRL)

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-04-31
"""

import os

from imodels import BayesianRuleListClassifier
from imodels.discretization import ExtraBasicDiscretizer
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split

from ante_hoc.plots.ante_hoc_plots import viz_classification_preds

# [1.] Import data + preprocessing -------------------------------------------------------------------------------------
data = loadarff(os.path.join("data", "input_data", "diabetes_dataset", "dataset_37_diabetes.arff"))
data_np = np.array(list(map(lambda x: np.array(list(x)), data[0])))
X = data_np[:, :-1].astype('float32')
y_text = data_np[:, -1].astype('str')
y = (y_text == 'tested_positive').astype(int)  # labels 0-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75)  # split
feat_names = ["#Pregnant", "Glucose concentration test", "Blood pressure(mmHg)",
               "Triceps skin fold thickness(mm)",
               "2-Hour serum insulin (mu U/ml)", "Body mass index", "Diabetes pedigree function", "Age (years)"]


disc = ExtraBasicDiscretizer(feat_names[:3], n_bins=3, strategy='uniform')
X_train_brl_df = disc.fit_transform(pd.DataFrame(X_train[:, :3], columns=feat_names[:3]))
X_test_brl_df = disc.transform(pd.DataFrame(X_test[:, :3], columns=feat_names[:3]))

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
