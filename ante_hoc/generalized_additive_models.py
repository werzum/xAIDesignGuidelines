"""
    Generalized Additive Models (GAM)

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-04-31
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.gam.tests.test_penalized import df_autos

# [1.] Input data ------------------------------------------------------------------------------------------------------
x_spline = df_autos[['weight', 'hp']]

bs = BSplines(x_spline, df=[12, 10], degree=[3, 3])

# [3.] GAM -------------------------------------------------------------------------------------------------------------
alpha = np.array([21833888.8, 6460.38479])
gam_bs = GLMGam.from_formula('city_mpg ~ fuel + drive', data=df_autos, smoother=bs, alpha=alpha)
res_bs = gam_bs.fit()

print(res_bs.summary())
