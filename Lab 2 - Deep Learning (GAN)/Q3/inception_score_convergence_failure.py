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

# this function calculate the inception score, and the input p_yx is a list of conditional probabilities
def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score

# generate images and calculate inception score for these images
def calculate_inception_score_generated_img(num_imgs_each_digit, n_split = 10, eps = 1E-16):
    noise = np.random.normal(0, 1, (num_imgs_each_digit * 10, latent_dim))
    tmp = []
    for i in range(num_imgs_each_digit):
        for digit in range(10):
            tmp.append(digit)
    sampled_labels = np.array(tmp)
    gen_imgs = generator.predict([noise, sampled_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    pred_list = []
    for i in range(gen_imgs.shape[0]):
        pred = model.predict(gen_imgs[i,:,:].reshape(1, 28, 28, 1))
        pred_list.append(pred[0])
        # pred[0] = [1.0000000e+00 3.7553105e-20 1.3926897e-13 1.4102576e-16 1.2462111e-23 1.8829189e-09 6.9483339e-14 8.5432184e-15 1.6225489e-13 8.5705963e-17]
    n_part = floor(gen_imgs.shape[0]/n_split)
    scores = list()
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        p_yx = np.array(pred_list[ix_start:ix_end])
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # p_y.shape = (1,10)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # kl_d.shape =  (10, 10) = (num_img_each_digit, num_img_each_digit)
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # sum_kl_d: [0.00302802 0.00023824 0.00023874 0.00023875 0.00039427 0.00023862 0.00022565 0.00023879 0.0002121  0.0002388 ]
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # avg_kl_d:  0.0005291983
        # undo the log
        is_score = exp(avg_kl_d)
        # is_score:  1.0005293
        # store
        scores.append(is_score)
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

is_avg, is_std = calculate_inception_score_generated_img(1000)
print("score is ", is_avg, ", ", is_std)




