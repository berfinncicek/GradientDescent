#advanced gradient descent

import numpy as np
import pandas as pd

def cost_Func(x,Y,b,w):  # MSE
    m = len(Y)
    sse = 0 #error sum of squares

    for i in range(0,m):
        y_pred = b + w* x[i]
        y = Y[i]
        sse += (y_pred - y)**2

        mse = sse / (2*m)

    return mse


def update_weights(x,Y,b,w, learning_rate):
    m = len(Y)
    b_sum = 0
    w_sum = 0

    for i in range(0,m):
        y_pred = b + w * x[i]
        y  = Y[i]

        b_sum += (y_pred - y)
        w_sum += (y_pred - y)* x[i]

    b = b - (learning_rate * 1 / m *b_sum)
    w = w - (learning_rate * 1 / m * w_sum)

    return b, w

def train(x, y ,initial_b,initial_w, num_iter,learning_rate):
    print("starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost_Func(x, y, initial_b, initial_w, )))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iter):
        b, w = update_weights(x,y, b, w,  learning_rate)
        mse = cost_Func(x,y,b,w)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iter, b, w, cost_Func(x,y, b, w,)))
    return cost_history, b, w

df = pd.read_csv(r"C:\Users\berfi\YAZILIM\algorithms\advertising.csv")

x = df["radio"]
y = df["sales"]

initial_b = np.random.randn(1).item()
initial_w = np.random.randn(1).item()
learning_rate = 0.001
num_iters = 100000

cost_history, b, w = train(x, y, initial_b, initial_w,num_iters, learning_rate )










