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
from math import floor
import os
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Dropout, Input,  Reshape, multiply
from keras.layers import Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam, Adam, SGD
from keras.datasets import mnist
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Image shape information

img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
if len(X_train.shape) == 4:
    channels = X_train.shape[3]
else:
    channels = 1

img_shape = (img_rows, img_cols, channels)
num_classes = 10
latent_dim = 100
optimizer = Adam(lr=0.004, beta_1=0.95)  # default : learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07


def generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    """
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    """
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    model_input = multiply([noise, label_embedding])
    img = model(model_input)
    return Model([noise, label], img)


def discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=np.prod(img_shape)))
    model.add(LeakyReLU(alpha=0.2))

    """
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    """
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])
    validity = model(model_input)
    return Model([img, label], validity)


discriminator = discriminator()
discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

# Build the generator
generator = generator()
generator.load_weights("../Q2/saved_model_weights/version4/generator_weights_29000.h5")

# the classifier model
path_save_model = 'save_weight_classifier/version_1.h5'
model = tf.keras.models.load_model(path_save_model)