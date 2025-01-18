#basic gradient descent

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = 2 * np.random.rand(100,1) #inputs
y= 4 + 3* x + np.random.randn(100,1) #y=b+wxi !!

b = np.random.randn(1)
w = np.random.randn(1)

learning_Rate = 0.1
iteration = 500
m = len(x)

cost_Func=[]

for i in range(iteration):

    y_prediction = b + w*x #model func

    cost = (1 / (2 * m)) * np.sum((y_prediction - y) ** 2) #cost func for gradient descent /MSE
    cost_Func.append(cost)

    #q0 = b , q1 = w

    q0 = (1 / m) * np.sum(y_prediction - y) #gradient b
    q1 = (1 / m) * np.sum((y_prediction - y) * x) # gradient w

    #update rule

    b -= learning_Rate * q0
    w -= learning_Rate * q1


print(f" bias (b): {b[0]:.4f}")
print(f" weight (w): {w[0]:.4f}")

plt.plot(range(iteration), cost_Func)
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('decreasing cost function')
plt.show()

sorted_indices = np.argsort(x.flatten())
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]
y_pred_sorted = (w * x + b)[sorted_indices]

plt.scatter(x_sorted, y_sorted, color='blue', label='Data')
plt.plot(x_sorted, y_pred_sorted, color='red', label='Model')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Model vs. Data')
plt.show()
