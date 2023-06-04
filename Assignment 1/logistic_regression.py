import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def transform(y):
    new_y = []
    for i in range(len(y)):
        new_y.append([1 if y[i] == x else 0 for x in range(1, 9)])
    new_y = np.array(new_y)
    return new_y


def calc_cost(X, y, weights):
    H = sigmoid(X @ weights.T)
    loss = -np.sum(y * np.log(H) + (1 - y) * np.log(1 - H))
    return loss / len(X)


def sigmoid(z):
    result = 1 / (1 + np.exp(-z))
    return result


def gradient_descent(X, y, weights, learning_rate, maxit):
    train_cost = np.zeros(maxit)
    val_cost = np.zeros(maxit)
    best_weights = weights

    price = 1000

    for i in range(maxit):
        H = sigmoid(X @ weights.T)
        weights = weights - (learning_rate * ((H - y).T @ X)) / len(X)
        train_cost[i] = calc_cost(X, y, weights)
    return weights, train_cost


def load_data(df):
    x = df.iloc[:, 2:]
    ones = np.ones([len(x), 1])
    x = np.concatenate((ones, x), axis=1)  # N x d+1
    y = np.array(df[1]).reshape(-1, 1)
    y = transform(y)
    return x, y


def run_logistic_regression(train_df, val_df, test_df):
    # train data
    x_train, y_train = load_data(train_df)
    x_val, y_val = load_data(val_df)

    x_test = test_df.iloc[:, 1:]
    ones = np.ones([len(x_test), 1])
    x_test = np.concatenate((ones, x_test), axis=1)
    x_test = pd.DataFrame(x_test)

    weights = np.zeros([8, x_train.shape[1]])  # 1 x d+1

    optimum_weights, train_cost = gradient_descent(
        x_train, y_train, weights, 0.001, 1665
    )

    pred = x_test @ optimum_weights.T
    pred = sigmoid(pred)
    return pred
