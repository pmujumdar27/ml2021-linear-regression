import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import rmse, mae

np.random.seed(42)

N = 100
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

X['new_col'] = X[X.columns[0]]+X[X.columns[1]]

for fit_intercept in [True, False]:
    if fit_intercept:
        print("---------- Fit intercept ON ------------")
    else:
        print("------------ Fit intercept OFF -----------")
    LR = LinearRegression(fit_intercept=fit_intercept,)
    LR.fit_vectorised(X, y, batch_size=1, n_iter=100, lr=0.01) # here you can use fit_non_vectorised / fit_autograd methods
    # LR.fit_non_vectorised(X, y, batch_size=1, n_iter=100, lr=0.01)
    # LR.fit_autograd(X, y, batch_size=10)
    y_hat = LR.predict(X)

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
    # LR.plot_line_fit(X, y, None, None)