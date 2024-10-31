# dmbreg (Dumb Regressors)
dmbreg offers a set of regression models of varying complexity designed for linear regression problems. The library implements two main types of regressors:

- "Dumb" regressors — use analytical solutions where possible and direct numerical minimizations when an analytical approach is not available. These regressors provide basic performance for standard regression tasks at reasonable resource costs.

- "Half Dumb" regressors are models using `scipy.optimize.minimize` numerical minimization algorithms to improve performance and accuracy in more complex cases.

# Table of Contents

- [Dependencies](#dependencies)
- [Models](#models)
- [Usage](#usage)
- [Installation and uninstallation](#installation-and-uninstallation)
- [License](#license)
- [Contacts](#contacts)

# Dependencies:

- numpy - for vector calculations.
- scipy - for numerical optimizations.

To satisfy dependencies, in the console (or other environment) use the command:
```console
pip install numpy scipy
```

# Models:
The following models are presented in both versions ("Dumb" and "Half Dumb"):

## Linear Regression

- class DumbLinearRegressor - a class that implements the analytical calculation of model coefficients using the inverse matrix, if the svd argument is not specified as True. If specified, implements the pseudo-inverse matrix by direct singular decomposition of the feature matrix.
- class HalfDumbLinearRegressor - a class that implements the analytical calculation of model coefficients using a pseudo-inverse matrix by means of numpy.

## Quantile Regression

- class DumbQuantileRegressor - a class that implements linear quantile regression. The loss function is minimized by manual gradient descent (numerical method).
- class HalfDumbQuantileRegressor - a class that implements linear quantile regression with the corresponding loss function. Minimization of the loss function is carried out by numerical means `scipy.optimize.minimize`.

## Lasso Regression

- DumbLassoRegressor - a class implementing linear lasso regression (with L1 regularization). The loss function is minimized by manual coordinate descent (numerical method).
- HalfDumbLassoRegressor - a class that implements linear lasso regression (with L1 regularization), minimization of the loss function is done by numerical means `scipy.optimize.minimize`.

## Ridge Regression

- DumbRidgeRegressor - a class that implements linear Ridge regression (with L2 regularization). The minimization of the loss function of the linear Ridge regression can be performed analytically, so in this case, the coefficients of the model are calculated analytically. A pseudo-inverse matrix is ​​used by means of numpy.
- HalfDumbRidgeRegressor - a class that implements linear Ridge regression (with L2 regularization). In this case, numerical minimization is performed using `scipy.optimize.minimize`.

## Elastic Net Regression

- DumbElasticNetRegressor - a class that implements linear ElasticNet regression (combination of L1 and L2 regularizations). The minimization of the loss function of such a regression cannot be performed analytically, therefore a manual implementation of the coordinate descent method is used.
- HalfDumbElasticNetRegressor - a class that implements linear ElasticNet regression (combination of L1 and L2 regularizations). In this case, numerical minimization is performed using `scipy.optimize.minimize`.

# Usage
## Model initialization and training

To train the model on the test data set, initialize an object of the regressor class and pass to the `fit` method:
```python
regressor = DumbQuantileRegressor()
regressor.fit(X_train,Y_train,learning_rate=0.01,epochs=10000,quantile=0.5,patience=100)
```
## Loss history array

If this is a "dumb" regressor and it was learned iteratively, then you can get the history of minimizing the loss function (array of values ​​of the loss function for each epoch):
```python
regressor.get_loss_history()
```

## Model coefficients
If you need to get an array of fitted coefficients, you can use:
```python
regressor.get_coefficients()
```

## Prediction
To predict values ​​based on a new set of features, after training, you can use:
```python
regressor.predict(X)
```

## Score
The regressor class allows you to calculate the $R^2$-coefficient of determination for training or test samples:
```python
regressor.score(X_train,Y_train)
regressor.score(X_test,Y_test)
```

# Installation and uninstallation
Available to install via:
```bash
pip install .
```
in the directory with the `setup.py` file.

Uninstallation is available via:
```bash
pip uninstall dmbreg
```

# License
MIT License

Copyright (c) 2024 [WhatsWrongHere]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# Contacts
Contact

Zabora Daniil
- e-mail [zaboradaniil@gmail.com]

