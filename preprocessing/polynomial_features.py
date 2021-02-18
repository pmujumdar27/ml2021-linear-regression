''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=True):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        self.degree = degree
        self.include_bias = include_bias

    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        dim_1 = False
        if len(X.shape)==1:
            # converting 1D array to 2D (to get features in columns)
            dim_1 = True
            X = np.array([list(X)])

        X_df = pd.DataFrame(X)

        # print(X)
        # print(X_df)

        new_X_df = pd.DataFrame()

        
        if self.include_bias:
            for col in list(X_df.columns):
                new_X_df[0] = X_df[col]**0

        cnt = 1
        for deg in range(1, self.degree+1):
            for col in list(X_df.columns):
                cnt+=1
                new_X_df[cnt] = X_df[col]**deg

        if dim_1:
            return np.array(new_X_df)[0]
        return np.array(new_X_df)
