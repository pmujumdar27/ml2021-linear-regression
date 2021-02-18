
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for fit_intercept in [True, False]:
    if fit_intercept:
        print('[config] Fit intercept ON')
    else:
        print('[config] Fit intercept OFF')
    LR = LinearRegression(fit_intercept=fit_intercept,)

    # LR.fit_non_vectorised(X, y, batch_size=1, n_iter=50, lr=0.01)
    # LR.fit_non_vectorised(X, y, batch_size=10, n_iter=50, lr=5, lr_type='inverse')

    # LR.fit_vectorised(X, y, batch_size=10, n_iter=50, lr=0.01)
    # LR.fit_vectorised(X, y, batch_size=10, n_iter=50, lr=5, lr_type='inverse')

    # LR.fit_autograd(X, y, batch_size=10, n_iter=50, lr=0.01)
    LR.fit_autograd(X, y, batch_size=10, n_iter=50, lr=5, lr_type='inverse')

    # LR.fit_normal(X, y)

    y_hat = LR.predict(X)

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
    print()
