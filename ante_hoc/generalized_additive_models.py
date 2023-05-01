"""
    Generalized Additive Models (GAM)

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-04-30
"""

import numpy as np
from pygam import LinearGAM, s, f

from plots.gam_plots import plot_gam_partial_dependence_functions
from preprocessing.input_data_preprocessing import preprocess_input_data
from sklearn.metrics import accuracy_score, confusion_matrix, mutual_info_score

from plots.confusion_matrix_plot import confusion_matrix_plot_sns

# [1.] Import data -----------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test, feature_names, target_names = preprocess_input_data()
features_nr = len(feature_names)
print("---------------------------------------------------------------------------------------------------------------")

# [2.] Grid search for 4 splines to each of the 4 features -------------------------------------------------------------
splines_nr = 20
lams = np.logspace(-5, 5, 100) * features_nr
gam = LinearGAM(s(0, n_splines=splines_nr) +
                s(1, n_splines=splines_nr) +
                s(2, n_splines=splines_nr) +
                s(3, n_splines=splines_nr)).gridsearch(
                X_train,
                y_train,
                lam=lams)
print(gam.summary())

print(f"GAM lam: {gam.lam}")
print("---------------------------------------------------------------------------------------------------------------")

# [3.] Evaluation ------------------------------------------------------------------------------------------------------
y_pred_orig = gam.predict(X_test)
y_pred = [round(x) for x in y_pred_orig]

check_prediction = all(item in y_pred for item in [0, 1, 2])     # Check that all predicted classes are valid ----------
assert check_prediction, f"The predicted classes are invalid, they should be one of {[0, 1, 2]}"

# [3.] Classification metrics ------------------------------------------------------------------------------------------
# [3.1.] Accuracy ------------------------------------------------------------------------------------------------------
acc_score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc_score}")

# [3.2.] Mutual Information (MI) ---------------------------------------------------------------------------------------
mi = mutual_info_score(y_test, y_pred)
print(f"Mutual Info: {mi}")

# [3.3.] Confusion Matrix - Create an annotated version ----------------------------------------------------------------
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

confusion_matrix_plot_sns(conf_matrix, "generalized_additive_model", features_nr, target_names)

# [4.] Plot the GAM's partial dependence functions ---------------------------------------------------------------------
plot_gam_partial_dependence_functions(gam, feature_names)

