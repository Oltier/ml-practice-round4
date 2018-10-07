import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from excercises.T2ModelSelection import feature_mapping, empirical_risk, predict


def regularizedFit(X, y, l=0):
    N = X.shape[0]
    w_opt = 1 / N * np.dot(
        np.dot(np.linalg.inv(1 / N * np.dot(np.transpose(X), X) + l * np.eye(X.shape[1])), np.transpose(X)), y)
    return w_opt


def regularizedPolynomialRegression(x, y, l=0, degree=2):
    X = feature_mapping(x, degree)
    w_opt = regularizedFit(X, y, l)
    return w_opt


# specify the degree
degree = 6
# specify list of values for lambda to be considered
lambdas = [0, 0.01, 0.1, 0.5, 1, 2, 5]


def trainValErrorsRegularization(x_train, y_train, x_val, y_val, lambdas=[0], degree=2):
    # compute the optimal weight, training and validation error for each lambda
    w_opts = []
    training_errors = []
    validation_errors = []
    for l in lambdas:
        w_opt = regularizedPolynomialRegression(x_train, y_train, l, degree)
        w_opts.append(w_opt)
        training_errors.append(empirical_risk(feature_mapping(x_train, degree), y_train, w_opt))
        validation_errors.append(empirical_risk(feature_mapping(x_val, degree), y_val, w_opt))
    return w_opts, training_errors, validation_errors


df = pd.read_csv('BTC_ETH_round4.csv')
x = df.Bitcoin.values  # Bitcoin values
y = df.Ethereum.values  # Ethereum values

x_train = x[1::2]
x_val = x[0::2]
y_train = y[1::2]
y_val = y[0::2]

# compute the training and validation errors and display them
w_opts_reg, training_errors_reg, validation_errors_reg = trainValErrorsRegularization(x_train, y_train, x_val, y_val,
                                                                                      lambdas, degree=degree)
df_lambdas = pd.DataFrame(
    data={'Lambdas': lambdas, 'Training errors': training_errors_reg, 'Validation errors': validation_errors_reg})
display(df_lambdas)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121)
plt.scatter(x_train, y_train, color='black', label='Training data')
plt.scatter(x_val, y_val, color='red', label='Validation data')

for i in range(len(lambdas)):
    plt.plot(np.linspace(0, 1, 50), predict(feature_mapping(np.linspace(0, 1, 50), degree), w_opts_reg[i]),
             label='$ \lambda=%s$' % str(lambdas[i]))

plt.title(r'$\bf{Figure\ 4.}$ Regularized Polynomial Regression')
plt.xlabel('Bitcoin')
plt.ylabel('Ethereum')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.legend()

# plot the training and validation errors for the different values of lambda
ax = fig.add_subplot(122)
plt.plot(lambdas, training_errors_reg, color='black', label='Training error')
plt.plot(lambdas, validation_errors_reg, color='red', label='Validation error')

plt.title(r'$\bf{Figure\ 5.}$ Training and validation error for different lambdas')
plt.xlabel('Lambda')
plt.ylabel('Empirical error')
plt.xticks(lambdas)
plt.legend()
plt.show()
