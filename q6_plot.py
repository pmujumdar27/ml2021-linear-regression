import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

plt.style.use('seaborn')

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

x = x.reshape(60,1) #Converting 1D to 2D for matrix operations consistency
y = pd.Series(y)

max_degree = 10


sample_sizes = [12,24,36,48,60]

for ss in sample_sizes:
    degrees = [1,3,5,7,9]
    thetas = []

    for degree in degrees:
        # degrees.append(degree)

        pf = PolynomialFeatures(degree)
        x_poly = pf.transform(x[:ss])
        X = pd.DataFrame(x_poly)

        y_ss = y[:ss]

        LR = LinearRegression(fit_intercept=False)
        LR.fit_vectorised(X, y_ss, 30, n_iter=7, lr = 0.0001)

        curr_theta = LR.coef_
        tot_theta = np.linalg.norm(curr_theta)
        thetas.append(tot_theta)

    # print(len(thetas))
    plt.yscale('log')
    plt.plot(degrees,thetas, label='Sample size: '+ str(ss))
    plt.title('Magnitude of theta vs Degree of Polynomial Features')
    plt.xlabel('Degree')
    plt.ylabel('Magnitude of Theta')
    plt.legend()

plt.savefig('plots/q6')