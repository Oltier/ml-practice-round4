import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


def fit(x, y):
    # Calculate the optimal weight vector
    w_opt = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    return w_opt


def feature_mapping(x, degree):
    # compute the feature map for the given degree
    polynomial_features = np.column_stack([x ** d for d in range(0, degree)])
    return polynomial_features


def polynomialRegression(x, y, degree):
    X = feature_mapping(x, degree)
    w_opt = fit(X, y)
    return w_opt


def predict(X, w_opt):
    # predict the labels
    y_pred = np.dot(X, w_opt)
    return y_pred


def empirical_risk(X, y, w_opt):
    empirical_error = np.mean(np.power(np.subtract(np.dot(X, w_opt), y), 2))
    return empirical_error


df = pd.read_csv('BTC_ETH_round4.csv')
x = df.Bitcoin.values  # Bitcoin values
y = df.Ethereum.values  # Ethereum values

# Split the data into a training and validation set.
# Don't change this.
x_train = x[1::2]
x_val = x[0::2]
y_train = y[1::2]
y_val = y[0::2]

# the degrees we want to loop over
degrees = [1, 3, 4]


def trainValErrors(x_train, y_train, x_val, y_val, degrees):
    # compute the optimal weight, training and validation error for each degree in degrees
    w_opts = []
    training_errors = []
    validation_errors = []
    for d in degrees:
        w_opt = polynomialRegression(x_train, y_train, d)
        w_opts.append(w_opt)
        training_errors.append(empirical_risk(feature_mapping(x_train, d), y_train, w_opt))
        validation_errors.append(empirical_risk(feature_mapping(x_val, d), y_val, w_opt))
    return w_opts, training_errors, validation_errors


# compute the training and validation errors and display them
w_opts, training_errors, validation_errors = trainValErrors(x_train, y_train, x_val, y_val, degrees)
df_degrees = pd.DataFrame(data={'d': degrees, 'E_train': training_errors, 'E_val': validation_errors})
display(df_degrees)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121)

plt.scatter(x_train, y_train, color='black', label='Training data')
plt.scatter(x_val, y_val, color='red', label='Validation data')
for i in range(len(degrees)):
    plt.plot(np.linspace(0, 1, 50), predict(feature_mapping(np.linspace(0, 1, 50), degrees[i]), w_opts[i]),
             label='degree = %d' % degrees[i])

plt.title(r'$\bf{Figure\ 2.}$ Polynomial Regression for different degrees')
plt.xlabel('Bitcoin')
plt.ylabel('Ethereum')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.legend()

# plot the training and validation errors for the different degrees
ax = fig.add_subplot(122)
plt.plot(degrees, training_errors, color='black', label='Training error')
plt.plot(degrees, validation_errors, color='red', label='Validation error')

plt.title(r'$\bf{Figure\ 3.}$ Training and validation error for different degrees')
plt.ylabel('Empirical error')
plt.xlabel('Degree')
plt.xticks(degrees)
plt.legend()

plt.show()
