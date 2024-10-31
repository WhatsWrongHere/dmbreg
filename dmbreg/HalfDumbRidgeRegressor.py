import numpy as np
from scipy.optimize import minimize

class HalfDumbRidgeRegressor:
    def __init__(self):
        self.beta = None
    
    def __loss(self, beta, X, Y, lambd):
        K = len(Y)
        residuals = Y - X @ beta
        MSE = (residuals @ residuals)/(2*K)
        sq_beta_sum = beta @ beta
        return MSE + (lambd/2)*sq_beta_sum
    
    def fit(self, X : np.ndarray, 
            Y : np.ndarray, 
            lambd : float = 0) -> None:
        """
        Fit the ridge regression model to the data.

        Parameters
        ----------
        - X : np.ndarray:
            Matrix of input features.
        - Y : np.ndarray:
            Vector of target values.
        - lambd : float, optional:
            A hyperparameter of L2 regularization. The default is 0.

        Returns
        -------
        - None
        """
        if len(X) != len(Y):
            raise ValueError(f"length of X({len(X)}) must be equal to length of Y({len(Y)})")

        if Y.ndim != 1:
            raise ValueError("Y must be a vector")

        if lambd < 0:
            raise ValueError(f"lambda parameter must be within >=0, but lambda = {lambd} occured")
        
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        
        initial_beta = np.zeros(X.shape[1])
        self.beta = minimize(self.__loss, 
                             x0 = initial_beta,
                             args=(X, Y, lambd)).x
        
    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        Predict the output based on previous fitted data.
        
        Parameters
        ----------
        - X : np.ndarray:
            Matrix of input features.

        Returns
        -------
        - np.ndarray: vector of predicted values.
        """
        
        if self.beta is None:
            raise RuntimeError("model isn't fitted")
            
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        if (X.shape[1] + 1) != len(self.beta):
            raise ValueError(f"Number of train parameters need to be equal to the test ones: {X.shape[1]} != {len(self.beta) - 1}")

        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        
        return (X @ self.beta).flatten()
    
    def score(self, X : np.ndarray, Y : np.ndarray) -> float:
        """
        Calculate the R^2 determination coefficient.
        
        Parameters
        ----------
        - X : np.ndarray:
            Matrix of input features.
        - Y : np.ndarray:
            Vector of expected output.

        Returns
        -------
        - float: R^2 determination coefficient.
        """
        predictions = self.predict(X)
        ss_total = ((Y-Y.mean()) ** 2).sum()
        ss_residual = ((Y - predictions) ** 2).sum()
        return 1 - (ss_residual/ss_total)
    
    def get_coefficients(self) -> np.ndarray:
        """
        Return the fitted coefficients if the model is fitted.

        Return:
        - np.ndarray: Array of fitted coefficients.
        """
        if self.beta is None:
            raise RuntimeError("The model has not been fitted yet.")
        return self.beta