import numpy as np
from scipy.optimize import minimize

class HalfDumbLogisticRegressor:
    def __init__(self):
        self.beta = None
        self.__cached_proba = None

    def __loss(self, beta, X, Y, weights):
        probabilities = 1 / (np.exp(-(X @ beta)) + 1)
        eps = 1e-10
        probabilities = np.clip(probabilities, eps, 1 - eps)
        log_loss = Y * np.log(probabilities) + (1 - Y) * np.log(1 - probabilities)

        return -(log_loss @ weights)

    def fit(self, X : np.ndarray, Y : np.ndarray, weights_balanced : bool = False) -> None:
        """
        Fit the quantile regression model to the data.

        Parameters
        ----------
        - X : np.ndarray:
            Matrix of input features.
        - Y : np.ndarray:
            Vector of target classes.
        - weights_balanced : bool:
            Model will balance weights if weights_balanced = True else won't.

        Returns
        -------
        - None
        """
        if len(X) != len(Y):
            raise ValueError(f"length of X({len(X)}) must be equal to length of Y({len(Y)})")

        if Y.ndim != 1:
            raise ValueError("Y must be a vector")

        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError("Y must contain only 0s and 1s for binary classification")

        if X.ndim == 1:
            X = X.reshape(-1,1)

        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        initial_beta = np.zeros(X.shape[1])
        if weights_balanced:
          w1 = (len(Y) / (Y == 1).sum())/2
          w0 = (len(Y) / (Y == 0).sum())/2
          weights = np.where(Y == 1, w1, w0)
        else:
          weights = np.ones(len(Y))
        self.beta = minimize(self.__loss,
                             x0 = initial_beta,
                             args = (X,Y, weights)).x
        self.__cached_proba = None

    def predict(self, X : np.ndarray, threshold : float = 0.5) -> np.ndarray:
        """
        Predict the output based on previous fitted data.

        Parameters
        ----------
        - X : np.ndarray:
            Matrix of input features.

        - threshold : float, optional:
            Threshold to determine class membership by predicted probability (if p >= threshold, prognosed class is 1, else 0).
            Default threshold is 0.5.

        Returns
        -------
        - np.ndarray: vector of predicted classes.
        """

        if self.beta is None:
            raise RuntimeError("model isn't fitted")

        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be within [0,1], but threshold = {threshold} occured")

        if X.ndim == 1:
            X = X.reshape(-1,1)

        if (X.shape[1] + 1) != len(self.beta):
            raise ValueError(f"Number of train parameters need to be equal to the test ones: {X.shape[1]} != {len(self.beta) - 1}")

        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        probabilities = 1 / (np.exp(-(X @ self.beta)) + 1)
        self.__cached_proba = probabilities

        return (probabilities >= threshold).astype(int)

    def get_proba(self) -> np.ndarray:
        """
        Retrieve the most recent predicted probabilities.

        Returns
        -------
        - np.ndarray: vector of predicted probabilities.
        """
        if self.__cached_proba is None:
            raise RuntimeError("there wasn't any predictions")

        return self.__cached_proba

    def reclassify(self, threshold : float) -> np.ndarray:
        """
        Reclassify the most recent predicted probabilities to classes with new threshold.

        Parameters
        ----------
        - threshold : float:
            Threshold to determine class membership by predicted probability (if p >= threshold, prognosed class is 1, else 0.


        Returns
        -------
        - np.ndarray: vector of predicted classes.
        """
        if self.__cached_proba is None:
            raise RuntimeError("there wasn't any predictions")
        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be within [0,1], but threshold = {threshold} occured")

        return (self.__cached_proba >= threshold).astype(int)


    def accuracy(self, Y: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate the fraction of correct predictions out of all predictions made (for the most rescent predictions).

        Parameters
        ----------
        - Y : np.ndarray:
            Vector of test (correct) predictions.

        - threshold : float, optional:
            Threshold to determine class membership by predicted probability (if p >= threshold, prognosed class is 1, else 0).
            Default threshold is 0.5.

        Returns
        -------
        - float: accuracy.
        """

        if self.__cached_proba is None:
            raise RuntimeError("there wasn't any predictions")

        if len(self.__cached_proba) != len(Y):
            raise ValueError(f"length of cached predictions ({self.__cached_proba}) must be equal to length of Y({len(Y)})")

        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be within [0,1], but threshold = {threshold} occured")

        if Y.ndim != 1:
            raise ValueError("Y must be a vector")

        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError("Y must contain only 0s and 1s for binary classification")

        predictions = self.reclassify(threshold)
        return (predictions == Y).mean()

    def precision(self, Y: np.ndarray, threshold: float = 0.5, target_class: int = 1) -> float:
        """
        Calculate the precision of the model for the specified threshold and class.

        Parameters
        ----------
        - Y : np.ndarray:
            Vector of test (correct) predictions.

        - threshold : float, optional:
            Threshold to determine class membership by predicted probability (if p >= threshold, prognosed class is 1, else 0).
            Default threshold is 0.5.

        - target_class : int, optional
            The class (0 or 1) for which to calculate precision.
            Default is 1 (positive class).

        Returns
        -------
        - float: precision.
        """
        if self.__cached_proba is None:
            raise RuntimeError("there wasn't any predictions")

        if len(self.__cached_proba) != len(Y):
            raise ValueError(f"length of cached predictions ({self.__cached_proba}) must be equal to length of Y({len(Y)})")

        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be within [0,1], but threshold = {threshold} occured")

        if Y.ndim != 1:
            raise ValueError("Y must be a vector")

        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError("Y must contain only 0s and 1s for binary classification")

        if target_class not in {0, 1}:
             raise ValueError(f"target_class must be within 0 or 1, but target_class = {target_class} occured")

        predictions = self.reclassify(threshold)
        tp = ((predictions == target_class) & (Y == target_class)).sum()
        total = (predictions == target_class).sum()
        return tp / total if total > 0 else 0

    def recall(self, Y: np.ndarray, threshold: float = 0.5, target_class: int = 1) -> float:
        """
        Calculate the recall of the model for the specified threshold and class.

        Parameters
        ----------
        - Y : np.ndarray:
            Vector of test (correct) predictions.

        - threshold : float, optional:
            Threshold to determine class membership by predicted probability (if p >= threshold, prognosed class is 1, else 0).
            Default threshold is 0.5.

        - target_class : int, optional
            The class (0 or 1) for which to calculate precision.
            Default is 1 (positive class).

        Returns
        -------
        - float: precision.
        """

        if self.__cached_proba is None:
            raise RuntimeError("there wasn't any predictions")

        if len(self.__cached_proba) != len(Y):
            raise ValueError(f"length of cached predictions ({self.__cached_proba}) must be equal to length of Y({len(Y)})")

        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be within [0,1], but threshold = {threshold} occured")

        if Y.ndim != 1:
            raise ValueError("Y must be a vector")

        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError("Y must contain only 0s and 1s for binary classification")

        if target_class not in {0, 1}:
             raise ValueError(f"target_class must be within 0 or 1, but target_class = {target_class} occured")

        predictions = self.reclassify(threshold)
        tp = ((predictions == target_class) & (Y == target_class)).sum()
        total = (Y == target_class).sum()
        return tp / total if total > 0 else 0

    def f1_score(self, Y: np.ndarray, threshold: float = 0.5, target_class: int = 1) -> float:
        """
        Calculate the f1 score of the model for the specified threshold and class.

        Parameters
        ----------
        - Y : np.ndarray:
            Vector of test (correct) predictions.

        - threshold : float, optional:
            Threshold to determine class membership by predicted probability (if p >= threshold, prognosed class is 1, else 0).
            Default threshold is 0.5.

        - target_class : int, optional
            The class (0 or 1) for which to calculate precision.
            Default is 1 (positive class).

        Returns
        -------
        - float: f1 score.
        """

        if self.__cached_proba is None:
            raise RuntimeError("there wasn't any predictions")

        if len(self.__cached_proba) != len(Y):
            raise ValueError(f"length of cached predictions ({self.__cached_proba}) must be equal to length of Y({len(Y)})")

        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be within [0,1], but threshold = {threshold} occured")

        if Y.ndim != 1:
            raise ValueError("Y must be a vector")

        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError("Y must contain only 0s and 1s for binary classification")

        if target_class not in {0, 1}:
             raise ValueError(f"target_class must be within 0 or 1, but target_class = {target_class} occured")

        prec = self.precision(Y, threshold, target_class)
        rec = self.recall(Y, threshold, target_class)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    def support(self, Y: np.ndarray, target_class: int = 1) -> int:
        """
        Count the occurrences of the target_class in Y.

        Parameters
        ----------
        - Y : np.ndarray:
            Vector of test (correct) predictions.

        - target_class : int, optional
            The class (0 or 1) for which to calculate precision.
            Default is 1 (positive class).

        Returns
        -------
        - int: number of occurrences.
        """

        if Y.ndim != 1:
            raise ValueError("Y must be a vector")

        if target_class not in {0, 1}:
             raise ValueError(f"target_class must be within 0 or 1, but target_class = {target_class} occured")

        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError("Y must contain only 0s and 1s for binary classification")

        return (Y == target_class).sum()



    def get_coefficients(self) -> np.ndarray:
        """
        Return the fitted coefficients if the model is fitted.

        Return:
        - np.ndarray: Array of fitted coefficients.
        """
        if self.beta is None:
            raise RuntimeError("The model has not been fitted yet.")
        return self.beta

    def get_classification_report(self, Y : np.ndarray, threshold : float = 0.5) -> str:
        """
        Get the classification report based on most recent predicted probabilities.

        Parameters
        ----------
        - Y : np.ndarray:
            Vector of test (correct) predictions.

        - threshold : float, optional:
            Threshold to determine class membership by predicted probability (if p >= threshold, prognosed class is 1, else 0).
            Default threshold is 0.5.

        Returns
        -------
        - str: classification report.
        """
        return f"""   precision    recall  f1-score   support

0       {self.precision(Y, threshold,0):.2f}      {self.recall(Y, threshold,0):.2f}      {self.f1_score(Y, threshold,0):.2f}       {self.support(Y,0)}
1       {self.precision(Y, threshold,1):.2f}      {self.recall(Y, threshold,1):.2f}      {self.f1_score(Y, threshold,1):.2f}       {self.support(Y,1)}

accuracy: {self.accuracy(Y, threshold):.2f}
        """
    def confusion_matrix(self, Y : np.ndarray, threshold : float = 0.5) -> np.ndarray:
        """
        Get the confusion matrix based on most recent predicted probabilities.

        Parameters
        ----------
        - Y : np.ndarray:
            Vector of test (correct) predictions.

        - threshold : float, optional:
            Threshold to determine class membership by predicted probability (if p >= threshold, prognosed class is 1, else 0).
            Default threshold is 0.5.

        Returns
        -------
        - np.ndarray: confusion matrix.
        """
        if self.__cached_proba is None:
            raise RuntimeError("there wasn't any predictions")

        if len(self.__cached_proba) != len(Y):
            raise ValueError(f"length of cached predictions ({self.__cached_proba}) must be equal to length of Y({len(Y)})")

        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be within [0,1], but threshold = {threshold} occured")

        if Y.ndim != 1:
            raise ValueError("Y must be a vector")

        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError("Y must contain only 0s and 1s for binary classification")

        predictions = self.reclassify(threshold)
        tp = ((predictions == 1) & (Y == 1)).sum()
        fp = ((predictions == 1) & (Y == 0)).sum()
        tn = ((predictions == 0) & (Y == 0)).sum()
        fn = ((predictions == 0) & (Y == 1)).sum()
        return np.array([[tn, fp], [fn, tp]])


