import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_cost(X, y, weights, mult):
    H = X @ weights.T
    J = np.power(H - y, 2)
    E = np.power(weights, 2)
    total = (np.sum(J) + mult * np.sum(E)) / (len(X))
    return total


def calc_mse(X, y, weights):
    H = X @ weights.T
    J = np.power(H - y, 2)
    total = np.sum(J) / (len(X))
    return total


def calc_mae(X, y, weights):
    H = X @ weights.T
    total = np.sum(abs(H - y)) / (len(X))
    return total


def gradient_descent(X, y, weights, learning_rate, maxit, mult):
    train_cost = np.zeros(maxit)
    val_cost = np.zeros(maxit)

    for i in range(maxit):
        H = X @ weights.T
        weights[:, 1:] = weights[:, 1:] - (
            2
            * (learning_rate * ((H - y).T @ X[:, 1:]) + mult * weights[:, 1:])
            / len(X)
        )
        weights[:, 0] = weights[:, 0] - (
            2 * (learning_rate / len(X)) * ((H - y).T @ X[:, 0])
        )
        train_cost[i] = calc_cost(X, y, weights, mult)

    return weights, train_cost


def load_data(df):
    x = df.iloc[:, 2:]
    ones = np.ones([len(x), 1])
    x = np.concatenate((ones, x), axis=1)  # N x d+1
    y = np.array(df[1]).reshape(-1, 1)
    return x, y


def run_ridge(train_df, val_df, test_df):
    # train data
    x_train, y_train = load_data(train_df)
    x_val, y_val = load_data(val_df)

    x_test = test_df.iloc[:, 1:]
    ones = np.ones([len(x_test), 1])
    x_test = np.concatenate((ones, x_test), axis=1)
    x_test = pd.DataFrame(x_test)

    weights = np.zeros([1, x_train.shape[1]])  # 1 x d+1

    optimum_weights, train_cost = gradient_descent(
        x_train, y_train, weights, 0.001, 835, 5
    )

    pred = x_test @ optimum_weights.T
    return pred
