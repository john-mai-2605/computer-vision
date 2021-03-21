from __future__ import division, print_function
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
from numpy import std
from math import floor, ceil
import os
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Dropout, Input,  Reshape, multiply
from keras.layers import Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam, Adam, SGD
from keras.datasets import mnist
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--real', type = float, default=0, help="ratio of real images")
parser.add_argument('--fake', type = float, default=0, help="ratio of fake images")
parser.add_argument('--size', type = int, default=60000, help="train size")

args = parser.parse_args()
perc_real = args.real
perc_fake = args.fake
num_train = args.size

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train = X_train[:num_train]
y_train = y_train[:num_train]
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
img_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    zoom_range = [0.9, 1],
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    brightness_range=[0.5,1.0],
    fill_mode='nearest')

datagen.fit(X_train)

test_gen =  ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)
test_gen.fit(X_test)

train_generator = datagen.flow(X_train, y_train, batch_size=60)
test_generator = test_gen.flow(X_test, y_test, batch_size=60)
# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
def classifier():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=img_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    return model

clf = classifier()
clf.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = clf.fit_generator(generator=train_generator,
                    use_multiprocessing=True,
                    steps_per_epoch = len(train_generator) / 3,
                    epochs = 30,
                    workers=-1)

a = clf.evaluate_generator(generator=test_generator)
print(a)
# image_index = 4444
# plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
# pred = clf.predict(x_test[image_index].reshape(1, 28, 28, 1))
# print(clf.argmax())

existing_models = os.listdir('save_weight_classifier')
new_path_save_model = 'save_weight_classifier/aug/' + str(100*perc_real) + '_' + str(100*perc_fake) + '.h5'
clf.save(new_path_save_model)

# model1 = tf.keras.models.load_model(new_path_save_model)
# model1.evaluate(x_test,y_test)
# image_index = 4444
# pred1 = model1.predict(x_test[image_index].reshape(1, 28, 28, 1))
# print(pred1.argmax())
