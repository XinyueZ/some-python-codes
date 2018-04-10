import matplotlib.pyplot as plt
import numpy as np
from numpy import max as maxium
from numpy import sum as summary
from numpy import arange, exp, ones_like, vstack


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = exp(x - maxium(x))
    return exp_x / summary(exp_x, axis = 0)

scores = [3.0, 1.0, 0.2]
print(softmax(scores))
x = arange(-2.0, 6.0, 0.1)
scores = vstack([x, ones_like(x), 0.2 * ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
