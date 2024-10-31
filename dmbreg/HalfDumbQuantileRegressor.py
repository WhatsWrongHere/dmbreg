import numpy as np
from scipy.optimize import minimize

class HalfDumbQuantileRegressor:
    def __init__(self):
        self.beta = None
    
    def __loss(self, beta, X, Y, q):
        residuals = (Y - X @ beta)
        return np.where(residuals >= 0, q * residuals, (q-1) * residuals).sum()
    
    def fit(self, X : np.ndarray, Y : np.ndarray, quantile : float = 0.5) -> None:
        """
        Fit the quantile regression model to the data.

        Parameters
        ----------
        - X : np.ndarray:
            Matrix of input features.
        - Y : np.ndarray:
            Vector of target values.
        - quantile : float, optional:
            A hyperparameter of quantile regression that lies within the closed interval [0,1] and regulates the penalty applied to the loss function. When q=0.5, the loss function is symmetric. If q>0.5, the loss function incurs larger penalties for overestimations (larger predictions), while if q<0.5, it incurs larger penalties for underestimations (smaller predictions). The default is 0.5.
        
        Returns
        -------
        - None
        """
        if len(X) != len(Y):
            raise ValueError(f"length of X({len(X)}) must be equal to length of Y({len(Y)})")
        
        if Y.ndim != 1:
            raise ValueError("Y must be a vector")
        
        if not (0 <= quantile <= 1):
            raise ValueError(f"quantile must be within [0,1], but quantile = {quantile} occured")
            
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        
        initial_beta = np.zeros(X.shape[1])
        self.beta = minimize(self.__loss, 
                             x0 = initial_beta, 
                             args = (X,Y,quantile)).x
        
    
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

    def get_loss_history(self) -> np.ndarray:
        """
        Get loss history array

        Returns
        -------
        - np.ndarray: loss history.
        """

        if self.loss_history is None:
            raise RuntimeError("model isn't fitted")
    
        return self.loss_history[:self.epochs]
    
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

        
        