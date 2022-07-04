import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import wandb
from wandb.keras import WandbCallback

# hyperParameters
hyperParameter_defaults = dict(
    batch_size=100,
    epochs=10,
    loss="sparse_categorical_crossentropy",
    pooling="AVERAGE",
    filters=64,
    Batch_normalization=False,
    kernel_size=3,
    dropout=0.5,
    hiddenLayer=2,
    convLayer=4,
    weight_decay=0.001
)

wandb.init(project="MNIST", entity="nunu", config=hyperParameter_defaults)
config = wandb.config
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

model = tf.keras.models.Sequential()

if config['convLayer'] > 0:
    model.add(tf.keras.layers.Conv2D(filters=config['filters'], input_shape=(28, 28, 1),
                                     kernel_size=(config['kernel_size'], config['kernel_size']),
                                     padding='SAME', activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(config['weight_decay'])))
    if config['pooling'] == "MAX":
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
    elif config['pooling'] == "AVERAGE":
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))

for i in range(1, config['convLayer']):
    model.add(
        tf.keras.layers.Conv2D(kernel_size=(config['kernel_size'], config['kernel_size']),
                               filters=config['filters'] * pow(2, i),
                               padding='same', activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(config['weight_decay'])))

model.add(
    tf.keras.layers.Conv2D(kernel_size=(config['kernel_size'], config['kernel_size']),
                           filters=config['filters'] * pow(2, config['convLayer']),
                           padding='valid', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(config['weight_decay'])))
if config['pooling'] == "MAX":
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
elif config['pooling'] == "AVERAGE":
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
model.add(tf.keras.layers.Flatten())

if config['hiddenLayer'] > 0:
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(config['dropout']))

if config['hiddenLayer'] > 1:
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(config['dropout']))

if config['hiddenLayer'] > 2:
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(config['dropout']))


if config['hiddenLayer'] > 3:
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    if config['Batch_normalization']:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(config['dropout']))


if config['Batch_normalization']:
    model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              # loss='sparse_categorical_crossentropy',
              loss=config['loss'],
              metrics=['accuracy'])

model.fit(x_edge_canny, y_train,
          validation_data=(x_test, y_test),
          callbacks=[WandbCallback()],
          epochs=config['epochs'],
          batch_size=config['batch_size'])

model.evaluate(x_test, y_test, verbose=2)
