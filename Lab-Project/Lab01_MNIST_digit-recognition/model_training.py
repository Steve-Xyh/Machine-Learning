#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
* Project Name : Machine-Learning
* File Name    : model-training.py
* Description  : Handwrite digit recognition model training
* Create Time  : 2020-05-11 12:59:38
* Version      : 1.0
* Author       : Steve X
* GitHub       : https://github.com/Steve-Xyh
'''

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from run_test import show_result

# 载入MNIST数据集
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

# 将数据 reshape 为 (n_images，x_shape，y_shape，channels)
# 对于灰度图像，channels 设置为 1, RGB 图像设置为 3
# n_images 设置为 -1, 表示与训练集图像总数相同
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

# 数据转为 float32 型
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# 归一化
train_X = train_X / 255
test_X = test_X / 255

# 标签转为独热码
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)


# 构建模型
model = Sequential()

# 卷积层 1,激活层,最大值池化层
# 64 个卷积核, 大小为 3*3
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷积层2,激活层,最大值池化层
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 一维化, 输出
model.add(Flatten())
model.add(Dense(64))

model.add(Dense(10))
model.add(Activation('softmax'))

# 编译, 训练, 保存模型
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=5)
model.save('model.h5')

# 评估模型
test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)
print('Test loss', test_loss)
print('Test accuracy', test_acc)

# 显示预测结果
show_result(isInput=False)
