import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from tqdm import tqdm

plt.style.use('seaborn')

np.random.seed(42)

# ----------------------- Samples vs Fit time -------------------------
N = 1000
P = 5
x = np.random.randn(N, P)
y = pd.Series(np.random.randn(N))

sample_sizes = []
sample_vs_time_vector = []
sample_vs_time_normal = []


for num_samples in tqdm(range(10, x.shape[0]+1, 10)):
    sample_sizes.append(num_samples)
    LR = LinearRegression(fit_intercept=True)
    start = time.time()
    LR.fit_vectorised(pd.DataFrame(x[:num_samples]), y[:num_samples], 30, n_iter=10, lr=0.02)
    fit_time = time.time()-start
    sample_vs_time_vector.append(fit_time)
    del LR
    LR = LinearRegression(fit_intercept=True)
    start = time.time()
    LR.fit_normal(pd.DataFrame(x[:num_samples]), y[:num_samples])
    fit_time = time.time()-start
    sample_vs_time_normal.append(fit_time)
    del LR

plt.figure()
plt.plot(sample_sizes, sample_vs_time_vector, label="Vectorized")
plt.plot(sample_sizes, sample_vs_time_normal, label="Normal")
plt.xlabel("Number of Samples")
plt.ylabel("Fit time")
plt.title("Fit time vs Number of Samples")
plt.legend()

plt.savefig('plots/q8_samples')


# -------------------------- Features vs Fit time -------------------
N = 300
P = 500

x = np.random.randn(N, P)
y = pd.Series(np.random.randn(N))

feature_sizes = []
feature_vs_time_vector = []
feature_vs_time_normal = []

for num_features in tqdm(range(1,P,2)):
    feature_sizes.append(num_features)
    tmp = pd.DataFrame(x)
    x_train = tmp[tmp.columns[:num_features]]
    LR = LinearRegression(fit_intercept=True)
    start = time.time()
    LR.fit_vectorised(x_train, y, 30, n_iter=10, lr=0.02)
    fit_time = time.time()-start
    feature_vs_time_vector.append(fit_time)
    del LR
    LR = LinearRegression(fit_intercept=True)
    start = time.time()
    LR.fit_normal(x_train, y)
    fit_time = time.time()-start
    feature_vs_time_normal.append(fit_time)
    del LR

plt.figure()
plt.plot(feature_sizes, feature_vs_time_vector, label="Vectorized")
plt.plot(feature_sizes, feature_vs_time_normal, label="Normal")
plt.xlabel("Number of Features")
plt.ylabel("Fit time")
plt.title("Fit time vs Number of Features")
plt.legend()

plt.savefig('plots/q8_features')