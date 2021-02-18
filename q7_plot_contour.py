import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

# plt.style.use('seaborn')

x = np.random.randn(100,)
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

x = x.reshape(100,1) #Converting 1D to 2D for matrix operations consistency
y = pd.Series(y)

# --------------- FIT LR --------------------
LR = LinearRegression(fit_intercept=True)
print("[Progress] Fitting model")
LR.fit_vectorised(pd.DataFrame(x), y, 30, n_iter=10, lr=0.04)


# ---------------- SURFACE PLOT ------------------
print("[Progress] Creating surface plots...")
l = []
for i in LR.theta_history:
    l.append(i.reshape(-1))

l = pd.DataFrame(l)
LR.plot_surface(pd.DataFrame(x), y, l[0], l[1])

# ----------------- LINE PLOT ---------------------
print("[Progress] Creating line plots...")
LR.plot_line_fit(pd.DataFrame(x), y, 0, 1)


# ---------------- CONOUR PLOT ------------------
print("[Progress] Creating contour plots...")
l = []
for i in LR.theta_history:
    l.append(i.reshape(-1))

l = pd.DataFrame(l)
LR.plot_contour(pd.DataFrame(x), y, l[0], l[1])