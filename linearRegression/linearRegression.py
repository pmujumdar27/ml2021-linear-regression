import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
# Import Autograd modules here
import autograd.numpy as anp
from autograd import grad

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.theta_history = []


    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        num_samples = X.shape[0]
        if self.fit_intercept:
            X = pd.concat([pd.Series(np.ones(num_samples)),X],axis=1)

        num_features = X.shape[1]
        self.coef_ = np.zeros((num_features,1))
        theta = self.coef_
        curr_lr = lr

        for iter in range(1, n_iter+1):
            # print(theta)
            if lr_type!='constant':
                curr_lr = lr/iter

            for batch in range(0, num_samples, batch_size):
                X_batch = np.array(X.iloc[batch:batch+batch_size])
                y_batch = np.array(y.iloc[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                batch_gradient = np.zeros((num_features,1))

                for curr_sample in range(curr_sample_size):
                    # accumulating batch gradient by calculating for all the samples in the current batch
                    x_sample = X_batch[curr_sample]
                    y_sample = y_batch[curr_sample]

                    y_hat_sample = 0
                    for a,b in zip(x_sample, theta):
                        # dot product
                        y_hat_sample += a*b

                    for feat in range(num_features):
                        batch_gradient[feat] -= (y_sample-y_hat_sample)*x_sample[feat]

                for feat in range(num_features):
                    theta[feat] -= (1/curr_sample_size)*curr_lr*batch_gradient[feat]


        self.coef_ = theta

    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        num_samples = X.shape[0]
        
        if self.fit_intercept:
            X = pd.concat([pd.Series(np.ones(num_samples)),X],axis=1)
        num_features = X.shape[1]

        self.coef_ = np.zeros((num_features,1))
        theta = self.coef_
        curr_lr = lr

        for iter in range(1, n_iter+1):
            # print("Iteration: {}".format(iter), theta)

            self.theta_history.append(theta.copy())

            if lr_type!='constant':
                curr_lr = lr/iter

            for batch in range(0, num_samples, batch_size):
                X_batch = np.array(X.iloc[batch:batch+batch_size])
                y_batch = np.array(y.iloc[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                y_hat_batch = np.matmul(X_batch, theta)
                error_batch = y_hat_batch - y_batch

                # update coeffs
                theta -= (1/curr_sample_size)*curr_lr*np.matmul(X_batch.T, error_batch)
                

        self.coef_ = theta

    def anp_loss(self, X, y, theta):
        y_hat = anp.matmul(X,theta)
        err = y_hat-y
        # squared error (will normalize while updating theta)
        return anp.matmul(err.T, err)
        

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        num_samples = X.shape[0]
        
        if self.fit_intercept:
            X = pd.concat([pd.Series(anp.ones(num_samples)),X],axis=1)
        num_features = X.shape[1]

        self.coef_ = anp.zeros((num_features,1))
        theta = self.coef_
        curr_lr = lr

        loss_grad = grad(self.anp_loss, argnum=2)

        for iter in range(1, n_iter+1):
            # print(theta)
            if lr_type!='constant':
                curr_lr = lr/iter

            for batch in range(0, num_samples, batch_size):
                X_batch = anp.array(X.iloc[batch:batch+batch_size])
                y_batch = anp.array(y.iloc[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                y_hat_batch = anp.matmul(X_batch, theta)
                error_batch = y_hat_batch - y_batch

                # update coeffs
                theta -= (1/curr_sample_size)*curr_lr*loss_grad(X_batch, y_batch, theta)

        self.coef_ = theta

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''

        X = np.array(X)

        if self.fit_intercept:
            tmp = np.ones((X.shape[0], 1))
            X = np.concatenate((tmp, X), axis=1)

        y=np.array(y)

        XTX = np.matmul(X.T,X)
        XTY = np.matmul(X.T,y)
        theta = np.matmul(np.linalg.pinv(XTX),XTY) #theta = (XTX)-^1XTY

        self.coef_ = theta
        return

    def predict(self, X, th=None):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X_dash = X

        if self.fit_intercept:
            tmp = np.ones((X.shape[0], 1))
            X_dash = np.concatenate((tmp, X), axis=1)

        if th is None:
            pred = np.dot(X_dash, np.array(self.coef_))
        else:
            pred = np.dot(X_dash, np.array(th))
        return pd.Series(pred.reshape(-1))

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        theta_opt = self.coef_
        theta_opt = theta_opt.reshape(-1)

        def fun(t0, t1):
            l = []
            for m,b in zip(t0, t1):
                curr_theta = np.array([m, b]).reshape((2,1))
                curr_y_hat = self.predict(X, th=curr_theta)
                err = curr_y_hat-y
                l.append(np.sum(err.dot(err.T)))
            return np.array(l)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xx = np.arange(theta_opt[0]-5, theta_opt[0]+5, 0.1)
        yy = np.arange(theta_opt[1]-5, theta_opt[1]+5, 0.1)
        X_grid, Y_grid = np.meshgrid(xx, yy)
        
        zz = np.array(fun(np.ravel(X_grid), np.ravel(Y_grid)))
        Z_grid = zz.reshape(X_grid.shape)

        z_points = fun(t_0, t_1)
        for i in range(len(t_0)-1):
            ax.scatter3D(np.array(t_0[:i+1]), np.array(t_1[:i+1]), z_points[:i+1], color='red', marker='x')
            ax.set_xlabel('t_0')
            ax.set_ylabel('t_1')
            ax.set_zlabel('RSS')
            ax.set_title('RSS: {}'.format(z_points[i]))
            
            ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.1)        

            plt.savefig('plots/surface/iter_{}'.format(i))
        

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """

        # y_hat = self.predict(X, self.theta_history[0])
        iter = 0
        for th in self.theta_history:
            iter+=1
            # print(th)
            plt.figure(figsize=(16,9))
            plt.title("t_0={} and t_1={}".format(th[0], th[1]))
            plt.xlabel("Iteration={}".format(iter))
            plt.scatter(X, y,  color='black')
            # th = np.array([th[t_0], th[t_1]])
            plt.xlim((min(X[0])-1, max(X[0])+1))
            plt.ylim(-2, max(y)+5)
            plt.plot(X, self.predict(X, th), color='blue', linewidth=3)
            plt.savefig('plots/line/iter_{}'.format(iter))

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        theta_opt = self.coef_
        theta_opt = theta_opt.reshape(-1)

        def fun(t0, t1):
            l = []
            for m,b in zip(t0, t1):
                curr_theta = np.array([m, b]).reshape((2,1))
                curr_y_hat = self.predict(X, th=curr_theta)
                err = curr_y_hat-y
                l.append(np.sum(err.dot(err.T)))
            return np.array(l)

        fig, ax = plt.subplots(figsize = (16,9))
        # fig = plt.figure()
        # ax = fig.add_subplot(111)

        xx = np.arange(theta_opt[0]-7, theta_opt[0]+7, 0.1)
        yy = np.arange(theta_opt[1]-7, theta_opt[1]+7, 0.1)
        X_grid, Y_grid = np.meshgrid(xx, yy)
        
        zz = np.array(fun(np.ravel(X_grid), np.ravel(Y_grid)))
        Z_grid = zz.reshape(X_grid.shape)

        z_points = fun(t_0, t_1)

        for i in range(len(z_points)-1):
            ax.contour(X_grid, Y_grid, Z_grid, 100)
            ax.set_xlabel('t_0')
            ax.set_ylabel('t_1')
            # ax.set_zlabel('RSS')
            ax.set_title('RSS: {}'.format(z_points[i]))
            tmp_x = t_0[i]
            tmp_y = t_1[i]
            dx = t_0[i+1]-tmp_x
            dy = t_1[i+1]-tmp_y
            plt.arrow(tmp_x, tmp_y, dx, dy, width=0.1)
            plt.savefig('plots/contour/iter_{}'.format(i))
