# ES654-2020 Assignment 3

*Pushkar Mujumdar* - *18110132*

------

> In this question we created a dataset that suffers from multicolinearity and tested our Linear Regression on that. The insights are stated below:

- We created a dataset with multicolinearity by adding a feature which is a linear combination of other 2 features.

The results for vectorised gradient descent are:
```
---------- Fit intercept ON ------------
RMSE:  0.8993670762910163
MAE:  0.7102067267837947

------------ Fit intercept OFF -----------
RMSE:  0.9110358255303003
MAE:  0.7169379745239525
```

As we can see, our model is pretty accurate on this dataset as well.

### NOTE: We cannot run normal equation implementation on this dataset because the matrix XtX is not invertible.