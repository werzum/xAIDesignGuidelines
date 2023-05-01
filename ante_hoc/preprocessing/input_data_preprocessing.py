"""
    Input data preprocessing

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-05-01
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def preprocess_input_data() -> tuple:
    """
    Preprocess the pre-defined input data

    :return: Tuple (could also be dict) with (X_train, X_test, y_train, y_test, feature_names, target_names)
    """

    # [1.] Import data -------------------------------------------------------------------------------------------------
    iris_data = load_iris()

    feature_names = iris_data.feature_names
    features_nr = len(feature_names)

    target_names = iris_data.target_names
    print(f"Feature names: {feature_names}, Nr. of features: {features_nr}")
    print(f"Target names: {target_names}")

    X = iris_data.data
    y = iris_data.target

    # [2.] Stratified split --------------------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test, feature_names, target_names
