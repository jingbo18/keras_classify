#!/usr/bin/env python
# coding: utf-8
# mnist神经网络训练，采用LeNet-5模型

import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adadelta, Adagrad

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.datasets import fashion_mnist
import h5py
from keras.models import model_from_json
import keras.backend as K

from utils import *


def channel_adaptor(input_data):
    print("input_data.shape:", input_data.shape)
    b, h, w = input_data.shape
    if K.image_data_format() == 'channels_first':
        # 单通道灰度图像,channel=1
        input_data = input_data.reshape(b, 1, h, w)
    else:
        input_data = input_data.reshape(b, h, w, 1)
    return input_data


def get_data(mnist):
    # 从图片文件加载数据
    (trainData, trainLabels), (testData, testLabels) = mnist.load_data()
    print("trainData shape:", trainData.shape)  # (60000, 28, 28)
    print("trainLabels shape:", trainLabels.shape)  # (60000,)
    print("trainData[0].shape:", trainData[0].shape)  # (28, 28)
    print("testLabels shape:", testLabels.shape)  # (10000,)
    print("testLabels[:20]:", testLabels[:20])  # 显示前20个数据 [9 2 1 1 6 1 4 6 5 7 4 5 7 3 4 1 2 4 8 0]
    # save_data_to_png(trainData, "train_data")
    # save_data_to_png(testData, "test_data")

    # label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
    trainLabels = np_utils.to_categorical(trainLabels, 10)
    testLabels = np_utils.to_categorical(testLabels, 10)
    print("categorical trainLabels shape:", trainLabels.shape)  # (60000, 10)
    # tensorflow后端

    trainData = channel_adaptor(trainData)
    testData = channel_adaptor(testData)

    # trainData = trainData.reshape(trainData.shape[0], 28, 28, 1)
    # testData = testData.reshape(testData.shape[0], 28, 28, 1)
    print("trainData.shape[0]:", trainData.shape[0])
    print("reshape trainData shape:", trainData.shape)
    return (trainData, trainLabels), (testData, testLabels)


def save_model(model, model_json, model_weights):
    # 保存model
    json_string = model.to_json()
    open(model_json, 'w').write(json_string)
    model.save_weights(model_weights)


def save_model_png(model, file_name):
    # 输出模型图片
    plot_model(model, to_file=file_name, show_shapes=True, show_layer_names=False)


def load_model(model_json, model_weights):
    # 读取model
    if not (os.path.exists(model_json) and os.path.exists(model_weights)):
        return None

    model = model_from_json(open(model_json).read())
    model.load_weights(model_weights)
    # 训练CNN模型
    return model


def get_model(model_json, model_weights):
    model = load_model(model_json, model_weights)
    if None is not model:
        return model

    model = Sequential()

    # model.add(Conv2D(4, 5, 5, border_mode='valid',input_shape=(28,28,1)))
    # 第一个卷积层，4个卷积核，每个卷积核5*5,卷积后24*24，第一个卷积核要申明input_shape(通道，大小) ,激活函数采用“tanh”
    model.add(Conv2D(filters=4, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), activation='tanh'))

    # model.add(Conv2D(8, 3, 3, subsample=(2,2), border_mode='valid'))
    # 第二个卷积层，8个卷积核，不需要申明上一个卷积留下来的特征map，会自动识别，下采样层为2*2,卷完且采样后是11*11
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='valid', activation='tanh'))
    # model.add(Activation('tanh'))

    # model.add(Conv2D(16, 3, 3, subsample=(2,2), border_mode='valid'))
    # 第三个卷积层，16个卷积核，下采样层为2*2,卷完采样后是4*4
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Activation('tanh'))

    model.add(Flatten())
    # 把多维的模型压平为一维的，用在卷积层到全连接层的过度

    # model.add(Dense(128, input_dim=(16*4*4), init='normal'))
    # 全连接层，首层的需要指定输入维度16*4*4,128是输出维度，默认放第一位
    model.add(Dense(128, activation='tanh'))

    # model.add(Activation('tanh'))

    # model.add(Dense(10, input_dim= 128, init='normal'))
    # 第二层全连接层，其实不需要指定输入维度，输出为10维，因为是10类
    model.add(Dense(10, activation='softmax'))
    # model.add(Activation('softmax'))
    # 激活函数“softmax”，用于分类

    return model


def model_compile(model):
    # 训练CNN模型
    sgd = SGD(lr=0.05, momentum=0.9, decay=1e-6, nesterov=True)
    # 采用随机梯度下降法，学习率初始值0.05,动量参数为0.9,学习率衰减值为1e-6,确定使用Nesterov动量
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # 配置模型学习过程，目标函数为categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列，第18行已转化，优化器为sgd
    return model


def train(model, trainData, trainLabels):
    # 训练模型，训练nb_epoch次，bctch_size为梯度下降时每个batch包含的样本数，验证集比例0.2,verbose为显示日志，shuffle是否打乱输入样本的顺序
    model.fit(trainData, trainLabels, batch_size=100, epochs=50, shuffle=True, verbose=1, validation_split=0.2)


def evaluate(model, testData, testLabels):
    # 对测试数据进行测试
    print(model.evaluate(testData, testLabels,
                         verbose=0,
                         batch_size=500))


def predict(model, x):
    h, w, d = x.shape
    return model.predict(x.reshape(-1, h, w, d))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用GPU:0
    mnist = fashion_mnist
    (trainData, trainLabels), (testData, testLabels) = get_data(mnist)
    model_json = 'my_model_architecture.json'
    model_weights = 'my_model_weights.h5'
    model = get_model(model_json, model_weights)
    save_model_png(model, "LeNet-5_model.png")
    model_compile(model)
    train(model, trainData, trainLabels)
    evaluate(model, testData, testLabels)
    print("predict x:", predict(model, testData[0]), testLabels[0])
