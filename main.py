
from darkai.supervised import linear_regression
from darkai.backends import *


lr = linear_regression(default_backend())

x, y = [1, 2, 3, 4, 5], [3, 4, 2, 4, 5]
lr.train(x, y)

for i in x:
    j = lr.predict(i)
    print("x: ", i, ", y: ", j)

