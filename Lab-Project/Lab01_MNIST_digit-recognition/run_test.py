#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
* Project Name : Machine-Learning
* File Name    : run.py
* Description  : Run model of Digit Recognition
* Create Time  : 2020-05-12 16:34:05
* Version      : 1.0
* Author       : Steve X
* GitHub       : https://github.com/Steve-Xyh
'''

from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import random


# 载入MNIST数据集
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

# 将数据 reshape 为 (n_images，x_shape，y_shape，channels)
# 对于灰度图像，channels 设置为 1, RGB 图像设置为 3
# n_images 设置为 -1, 表示与训练集图像总数相同
test_X = test_X.reshape(-1, 28, 28, 1)

# 加载训练好的模型, 进行预测
model = load_model('model.h5')
predictions = model.predict(test_X)


def show_result(test_num=0, isInput=True):
    '''
    Show the result of test model
    #### Parameters::
        test_num - the index of test array
        isInput - whether `test_num` is input or not
    '''
    if not isInput:
        test_num = random.randint(0, 9999)
    print('-' * 50)
    print(
        f'The number of test_X[{test_num}] is:\t',
        np.argmax(np.round(predictions[test_num]))
    )

    plt.imshow(test_X[test_num].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()


if __name__ == "__main__":
    for i in range(10):
        test_num = random.randint(0, 9999)
        show_result(test_num=test_num)
    print('-' * 50)
