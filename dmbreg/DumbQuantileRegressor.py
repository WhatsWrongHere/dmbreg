import numpy as np

class DumbQuantileRegressor:
    def __init__(self):
        self.beta = None
        self.loss_history = None
        self.epochs = None
    
    def __loss(self, X, Y, q):
        residuals = Y - X @ self.beta
        return np.where(residuals >= 0, q * residuals, (q-1) * residuals).sum()
    
    def __grad(self, X, Y, q):
        residuals = Y - X @ self.beta
        weights = np.where(residuals >=0, -q, 1 - q)
        return (X.T @ weights)/len(X)
    
    
    def fit(self, X : np.ndarray, 
            Y : np.ndarray,
            learning_rate : float = 0.01,
            epochs : int = 1000,
            quantile : float = 0.5,
            patience : int = 1000) -> None:
        """
        Fit the quantile regression model to the data.

        Parameters
        ----------
        - X : np.ndarray:
            Matrix of input features.
        - Y : np.ndarray:
            Vector of target values.
        - learning_rate : float, optional:
            Hyperparameter that controls the step size taken during each iteration of gradient descent optimization. The default is 0.01.
        - epochs : int, optional:
            Max number of iterations during the gradient descent minimization of the loss function. The default is 1000.
        - quantile : float, optional:
            A hyperparameter of quantile regression that lies within the closed interval [0,1] and regulates the penalty applied to the loss function. When q=0.5, the loss function is symmetric. If q>0.5, the loss function incurs larger penalties for overestimations (larger predictions), while if q<0.5, it incurs larger penalties for underestimations (smaller predictions). The default is 0.5.
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
            
        if learning_rate <= 0: 
            raise ValueError(f"learning_rate must be > 0, but learning_rate = {learning_rate} occured")
        
        if epochs < 1:
            raise ValueError(f"Number of epochs must be >= 1, but epochs = {epochs} occured")
            
        if not (0 <= quantile <= 1):
            raise ValueError(f"quantile must be within [0,1], but quantile = {quantile} occured")
            
        if patience < 1:
            raise ValueError(f"patience must be >=1, but patience = {patience} occured")
        
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        
        self.beta = np.zeros(X.shape[1])
        self.epochs = epochs
        self.loss_history = np.empty(epochs, dtype = float)
        
        best_metric = self.__loss(X, Y, quantile)
        patience_calc = 0
        for epoch in range(epochs):
            self.beta -= learning_rate * self.__grad(X, Y, quantile)
            loss = self.__loss(X, Y, quantile)
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
    
