"""
    Generalized Additive Models (GAM) plots

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-05-01
"""

import os

import matplotlib.pyplot as plt


def plot_gam_partial_dependence_functions(gam, feature_names: list):
    """
    Plot the GAM's partial dependence functions

    :param gam: Generalized Additive Model
    :param feature_names: List of feature names

    :return:
    """

    output_data_path = os.path.join(os.path.join("data", "output_data", "plots", "generalized_additive_model"))

    for i, term in enumerate(gam.terms):

        if term.isintercept:
            continue

        fig = plt.figure(figsize=(10, 10))
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

        plt.plot(XX[:, term.feature], pdep)
        plt.plot(XX[:, term.feature], confi, c='r', ls='--')
        plt.title(f"{repr(term)}: {feature_names[i]}")
        plt.xlabel("feature value")
        plt.ylabel("predicted value")
        plt.show()
        fig.savefig(os.path.join(output_data_path, f"gam_{feature_names[i]}.png"))
        plt.close()
