import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import wandb
from wandb.keras import WandbCallback


def prewitt(img):
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_d = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])

    prewitt_filter_x = cv2.filter2D(img, -1, prewitt_x)
    prewitt_filter_y = cv2.filter2D(img, -1, prewitt_y)
    prewitt_filter_d = cv2.filter2D(img, -1, prewitt_d)

    return prewitt_filter_x, prewitt_filter_y, prewitt_filter_d


def scharr(img):
    scharrX = cv2.Sobel(img, -1, 1, 0, ksize=1)
    scharrY = cv2.Sobel(img, -1, 0, 1, ksize=1)
    scharr = scharrY + scharrX
    return scharr


# 주문 : Edge Detector 구조, Feature map filter(=conv filter), hidden layer의 크기
# Polling(Max,Average), Activation function, Batch normalization, Dropout
# https://keras.io/ko/layers/convolutional/ conv 종류가 좀 있음

# hyperParameters
hyperParameter_defaults = dict(
    batch_size=100,
    epochs=15,
    loss="sparse_categorical_crossentropy",
    pooling="AVERAGE",
    filters=1,
    Batch_normalization=True,
    kernel_size=3,
    dropout=0.2,
    Activation="Both",
    metrics="accuracy",
    hiddenLayer=2
)

# wandb.init(project="MNIST", entity="nunu", config=hyperParameter_defaults)
# config = wandb.config
config = hyperParameter_defaults
normalize = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_data_size = x_train.shape
img_width, img_height = train_data_size[1], train_data_size[2]  # 28,28
train_shape = (img_width, img_height, 1)  # 28,28,1

# 명시적 채널 차원 설정 (Conv2D function 사용 위함)
x_train = x_train.reshape(len(x_train), train_shape[0], train_shape[1], train_shape[2])
x_test = x_test.reshape(len(x_test), train_shape[0], train_shape[1], train_shape[2])

x_edge_canny = np.array([cv2.Canny(x_train[i], 100, 255) for i in range(x_train.shape[0])])
"""
x_edge_laplacian = np.array([cv2.Laplacian(x_train[i], cv2.CV_8U, ksize=3) for i in range(x_train.shape[0])])
x_edge_sobel = np.array([cv2.Sobel(x_train[i], cv2.CV_8U, 1, 1, 1) for i in range(x_train.shape[0])])
x_edge_prewitt = np.array([prewitt(x_train[i])[2] for i in range(x_train.shape[0])])
x_edge_scharr = np.array([scharr(x_train[i]) for i in range(x_train.shape[0])])
# x_edge_canny = np.array([cv2.Canny(x_train[i], 100, 255) for i in range(1)])
# x_edge_laplacian = np.array([cv2.Laplacian(x_train[i], cv2.CV_8U, ksize=3) for i in range(1)])
# x_edge_sobel = np.array([cv2.Sobel(x_train[i],cv2.CV_8U, 1, 0, 5) for i in range(1)])


# cv2.imshow("sobel", x_edge_sobel[0])
# cv2.imshow("canny", x_edge_canny[0])
# cv2.imshow("laplacian", x_edge_laplacian[0])

cv2.imwrite('sobel.png', x_edge_sobel[0])
cv2.imwrite('canny.png', x_edge_canny[0])
cv2.imwrite('laplacian.png', x_edge_laplacian[0])
cv2.imwrite('prewitt.png', x_edge_prewitt[0])
cv2.imwrite('scharr.png', x_edge_scharr[0])
cv2.imwrite('train.png', x_train[0])

# cv2.waitKey(0)
# cv2.destroyAllWindows()
"""

# "mse"


# ... Define a model
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(config['filters'], input_shape=(28, 28, 1), kernel_size=(3, 3), padding='SAME'),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(input_shape=(14, 14)),
    # tf.keras.layers.Flatten(input_shape=(28, 28,1)),
    tf.keras.layers.Dense(196, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax'),
])
"""
model = tf.keras.models.Sequential()
if config['hiddenLayer'] > 0:
    model.add(tf.keras.layers.Conv2D(config['filters'], input_shape=(28, 28, 1),
                                     kernel_size=(config['kernel_size'], config['kernel_size']), padding='SAME'))
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    if config['pooling'] == "MAX":
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
    elif config['pooling'] == "AVERAGE":
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
if config['hiddenLayer'] > 1:
    model.add(tf.keras.layers.Conv2D(config['filters'], input_shape=(29-config['kernel_size'], 29-config['kernel_size'], 1),
                                     kernel_size=(config['kernel_size'], config['kernel_size']), padding='SAME'))
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    if config['pooling'] == "MAX":
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
    elif config['pooling'] == "AVERAGE":
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

if config['Activation'] == 'both' or config['Activation'] == 'relu':
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(196, activation='relu'))

if config['Batch_normalization']:
    model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(config['dropout']))

if config['Activation'] == 'both' or config['Activation'] == 'softmax':
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              # loss='sparse_categorical_crossentropy',
              loss=config['loss'],
              metrics=config['metrics'])
"""
model.fit(x_edge_canny, y_train,
          validation_data=(x_test, y_test),
          epochs=epochs,
          batch_size=batch_size)
"""
model.fit(x_edge_canny, y_train,
          validation_data=(x_test, y_test),
          # callbacks=[WandbCallback()],
          epochs=config['epochs'],
          batch_size=config['batch_size'])

model.evaluate(x_test, y_test, verbose=2)
