import numpy as np

class DumbElasticNetRegressor:
    def __init__(self):
        self.beta = None
        self.loss_history = None
        self.epochs = None
    
    def __loss(self, X, Y, lambda1, lambda2):
        K = len(Y)
        residuals = Y - X @ self.beta
        MSE = (residuals @ residuals)/(2*K)
        abs_beta_sum = np.abs(self.beta).sum()
        sq_beta_sum = (self.beta @ self.beta)
        return MSE + lambda1 * abs_beta_sum + (lambda2/2) * sq_beta_sum
    
    def __soft_treeshold(self, z, l):
        if (z > l):
            return z - l
        elif (z < -l):
            return z + l
        else:
            return 0
    
    def fit(self, X : np.ndarray,
            Y : np.ndarray,
            lambda1 : float = 0,
            lambda2 : float = 0,
            epochs : int = 1000,
            patience : int = 1000) -> None:
        """
        Fit the Elastic Net regression model to the data.

        Parameters
        ----------
        - X : np.ndarray:
            Matrix of input features.
        - Y : np.ndarray:
            Vector of target values.
        - lambda1 : float, optional:
            A hyperparameter of L1 regularization. The default is 0.
        - lambda2 : float, optional:
            A hyperparameter of L2 regularization. The default is 0.
        - epochs : int, optional:
            Max number of iterations during the minimization of the loss function. The default is 1000.
        - patience : int, optional:
            Max number of consecutive iterations when monitored metric may not decrease before triggering early stopping. The default is 1000.

        Returns
        -------
        - None
        """
        if len(X) != len(Y):
            raise ValueError(f"length of X({len(X)}) must be equal to length of Y({len(Y)})")

        if Y.ndim != 1:
            raise ValueError("Y must be a vector")
            
        if lambda1 < 0:
            raise ValueError(f"lambda parameter must be within >=0, but lambda = {lambda1} occured")

        if lambda2 < 0:
            raise ValueError(f"lambda parameter must be within >=0, but lambda = {lambda2} occured")

        if epochs < 1:
            raise ValueError(f"Number of epochs must be >= 1, but epochs = {epochs} occured")

        if patience < 1:
            raise ValueError(f"patience must be >=1, but patience = {patience} occured")
    
        if X.ndim == 1:
            X = X.reshape(-1,1)

        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        
        self.beta = np.zeros(X.shape[1])
        self.epochs = epochs
        self.loss_history = np.empty(epochs, dtype = float)
        
        best_metric = self.__loss(X, Y, lambda1, lambda2)
        patience_calc = 0
        K = len(Y)
        for epoch in range(epochs):
            for j in range(len(self.beta)):
                norm = ((X[:,j] @ X[:,j])/K) + lambda2
                rj = Y - X @ self.beta + self.beta[j] * X[:,j]
                z = (rj @ X[:,j])/K
                self.beta[j] = self.__soft_treeshold(z, lambda1)/norm
        
            loss = self.__loss(X, Y, lambda1, lambda2)
            self.loss_history[epoch] = loss
        
            if loss >= best_metric:
                patience_calc += 1
            else:
                best_metric = loss
                patience_calc = 0
            
            if((epoch + 1)%100 == 0):
                print(f"Epoch: {epoch+1}, Loss: {loss}")
      
            if(patience_calc == patience):
                self.epochs = epoch         # for get_loss_history meothod
                print(f"Early stopped on {epoch+1} epoch")
                break

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