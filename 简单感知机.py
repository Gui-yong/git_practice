import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)
# 设置线性可分的数据
data = pd.DataFrame(np.array([1, 3, 1, 1, 2, 1, 2, 3, 1, 2, 1, 0, 3, 1, 0, 4, 1, 0]).reshape((-1, 3)),
                    columns=["x1", "x2", "标签"])
# print(data)


class Perceptron:
    def __init__(self):
        self.w, self.b = self.init_parameter()

    def init_parameter(self):
        w = np.random.normal(0, 1, (data.shape[1] - 1, ))
        b = np.random.normal(0, 1, 1)
        return w, b


perceptron = Perceptron()

print(perceptron.w, perceptron.b)
