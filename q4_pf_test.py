import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures



# X = np.array([[1,2], [3,4]])
X = np.array([1,2])
for include_bias in [True, False]:
    poly = PolynomialFeatures(2, include_bias=include_bias)
    if include_bias:
        print('[Config] Include bias ON')
    else:
        print('[Config] Include bias OFF')
    print("Input: {}".format(X))
    print("Output: {}".format(poly.transform(X)))
    print()
    del poly
